/* fmt=100 (native FP8-e4m3 passthrough) loader-seam tests.
 *
 * fmt=100, not fmt=6: this format was minted fmt=6 during original development
 * of this branch, before dev's own #465 (E8/IQ3) claimed that ordinal upstream
 * and merged it into dev as a REAL fmt=6 (see quant.h's E8 constants and e8_
 * helper functions, and qt_resolve_fmt's ns==4-tag early check). Ported onto
 * dev post-#465-merge as fmt=100 from the start (PRIVATE
 * ORDINAL BLOCK, see colibri.c's QT struct comment) -- there was never a build
 * in this branch's new history where this format was reachable as fmt=6.
 *
 * Part A: qt_resolve_fmt disambiguation suite -- THE DESIGN LANDMINE. fmt=100
 * weight bytes are byte-identical to fmt=1 (int8): both are O*I raw bytes.
 * The two are told apart ONLY by the scale array's byte count (per-row O*4 for
 * fmt=1, per-128x128-block ceil(O/128)*ceil(I/128)*4 for fmt=100). For some
 * shapes those two counts coincide exactly -- qt_resolve_fmt must REFUSE
 * (exit(1)) rather than guess. Refusal is tested via fork()+waitpid(), mirroring
 * tests/test_st_pread.c's established convention for exit(1)-terminated paths.
 *
 * Part B: qt_from_disk loader-seam -- writes a real single-shard .safetensors
 * file containing an fmt=100 tensor (U8 weight + per-block F32 .qs) next to an
 * int8 control tensor of a DIFFERENT, non-colliding shape, loads both through
 * qt_from_disk, and checks the byte-count/.qs-size inference picks fmt=100 vs
 * fmt=1 correctly and the loaded weights dequantize identically to a reference.
 * Mirrors tests/test_int3_load.c's structure for fmt=5.
 *
 * NOTE (r2 rebase onto dev post-#465): dev's own fmt=6 (E8/IQ3) introduces a
 * SEPARATE collision at [O<=128, I=98] against fmt=100 that this suite, as
 * ported, does not yet cover -- see the reconciliation commit that follows
 * this port and test_fmt6_fp8_collision() it adds, plus the build report's
 * "THE SEMANTIC RECONCILIATION" section for the derivation and fail-before/
 * pass-after evidence. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/wait.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

static uint64_t rng = 0xFEEDFACE0DDBA11ull;
static float rndf(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return ((int64_t)(rng & 0xFFFFF) - 0x80000) / (float)0x80000; }
static uint8_t rndbyte_nonan(void){
    for(;;){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        uint8_t b = (uint8_t)(rng & 0xFF);
        if(b != 0x7F && b != 0xFF) return b; }
}

/* ---- Part A: qt_resolve_fmt disambiguation (in-process for non-refusing
 * cases, fork+waitpid for refusing ones -- qt_resolve_fmt exit(1)s in place,
 * it does not return an error code). ---- */

static int expect_fmt(int O, int I, int64_t nb, int64_t ns, int expect_fmt_val, const char *tag){
    int gs=0;
    int fmt = qt_resolve_fmt(tag, O, I, nb, ns, &gs);
    if(fmt != expect_fmt_val){
        printf("FAIL %s: got fmt=%d, expected fmt=%d (O=%d I=%d nb=%lld ns=%lld)\n",
               tag, fmt, expect_fmt_val, O, I, (long long)nb, (long long)ns);
        return 0;
    }
    return 1;
}

static int expect_refuse(int O, int I, int64_t nb, int64_t ns, const char *tag){
    int pipefd[2]; if(pipe(pipefd)!=0) return 0;
    pid_t pid = fork();
    if(pid < 0) return 0;
    if(pid == 0){
        dup2(pipefd[1],2); close(pipefd[0]); close(pipefd[1]);
        int gs=0;
        qt_resolve_fmt(tag, O, I, nb, ns, &gs);   /* must exit(1) inside; must NOT return */
        _exit(42);                                  /* reaching here is the bug */
    }
    close(pipefd[1]);
    char err[1024]={0}; ssize_t n=read(pipefd[0],err,sizeof(err)-1); (void)n;
    close(pipefd[0]);
    int status=0; waitpid(pid,&status,0);
    int ok = WIFEXITED(status) && WEXITSTATUS(status)==1;
    if(!ok){
        printf("FAIL %s: expected exit(1) refusal, got status=%d, stderr=%.200s\n", tag, status, err);
        return 0;
    }
    if(!strstr(err,"refus")){
        printf("FAIL %s: exited(1) but message lacked a refusal explanation: %.200s\n", tag, err);
        return 0;
    }
    return 1;
}

static void test_disambiguation(void){
    /* --- non-degenerate golden paths (unambiguous either way) --- */
    CHECK(expect_fmt(4096,4096,(int64_t)4096*4096,4096*4,1,"plain int8 4096x4096"));
    /* spec's own worked example: [2048,6144] expert -> block scale [16,48] */
    CHECK(expect_fmt(2048,6144,(int64_t)2048*6144,16LL*48*4,100,"fp8 2048x6144 (spec example)"));
    CHECK(expect_fmt(384,6144,(int64_t)384*6144,3LL*48*4,100,"fp8 384x6144 non-square block grid"));

    /* --- degenerate shapes: O<=128 makes nblkO==1, so ns_blk==nblkI*4 can
     * equal ns_row==O*4 whenever nblkI==O. Exhaustive-in-spirit sweep of the
     * boundary the design landmine describes. --- */
    CHECK(expect_refuse(1,1,      1,        4,  "degenerate O=1 I=1 (nblkI=1=O)"));
    CHECK(expect_refuse(1,128,    128,      4,  "degenerate O=1 I=128 (nblkI=1=O, I at block edge)"));
    CHECK(expect_refuse(2,256,    2LL*256,  8,  "degenerate O=2 I=256 (nblkI=2=O)"));
    CHECK(expect_refuse(6,768,    6LL*768,  24, "degenerate O=6 I=768 (nblkI=6=O)"));
    CHECK(expect_refuse(128,16384,128LL*16384, 512, "degenerate O=128 I=16384 (nblkO=1,nblkI=128=O)"));
    /* O>128 degenerate case: nblkO=2, need nblkI=O/2 -- O=256,I=16384 -> nblkI=128, 2*128=256=O */
    CHECK(expect_refuse(256,16384,256LL*16384, 1024, "degenerate O=256 I=16384 (nblkO=2,nblkI=128, product=O)"));

    /* --- boundary-ADJACENT non-degenerate cases: one step past each
     * degenerate case above, both interpretations now legitimately resolve. --- */
    CHECK(expect_fmt(1,129, 129,   4, 1, "adjacent O=1 I=129 as fmt=1 (ns=row)"));
    CHECK(expect_fmt(1,129, 129,   8, 100, "adjacent O=1 I=129 as fmt=100 (ns=block, nblkI=2)"));
    CHECK(expect_fmt(2,257, 2LL*257, 8,  1, "adjacent O=2 I=257 as fmt=1 (ns=row)"));
    CHECK(expect_fmt(2,257, 2LL*257, 12, 100, "adjacent O=2 I=257 as fmt=100 (ns=block, nblkI=3)"));

    /* --- neither interpretation matches: garbage .qs size, must still refuse
     * (the pre-existing generic-mismatch path, exercised through the fmt=100-aware
     * function to confirm the new code didn't disturb it). --- */
    CHECK(expect_refuse(10,10, 100, 999, "garbage ns matches neither row nor block layout"));
}

/* ---- Part B: qt_from_disk loader-seam (real safetensors file) ---- */

static void deq_fmt100(const QT *t, float *dq){
    int64_t nblkI = fp8_nblk(t->I);
    for(int o=0;o<t->O;o++){
        int64_t blkO = o/FP8_BLOCK; const float *scl = t->s + blkO*nblkI;
        for(int i=0;i<t->I;i++){
            int64_t bi = i/FP8_BLOCK;
            dq[(int64_t)o*t->I+i] = e4m3_decode(t->q8[(int64_t)o*t->I+i]) * scl[bi];
        }
    }
}
static void deq_fmt1(const QT *t, float *dq){
    for(int o=0;o<t->O;o++){ float s=t->s[o];
        for(int i=0;i<t->I;i++) dq[(int64_t)o*t->I+i]=(float)t->q8[(int64_t)o*t->I+i]*s; }
}

#define CDIV(n,d) (((n)+(d)-1)/(d))

static void test_loader_seam(void){
    enum { O100=8, I100=256 };                    /* nblkO=1, nblkI=2 -> 2 block scales total */
    enum { O1=5, I1=64 };                        /* DIFFERENT shape from the fp8 tensor: O1*I1=320
                                                   * bytes, ns=O1*4=20 -- neither collides with the
                                                   * fp8 tensor's own byte counts (kept deliberately
                                                   * distinct so this is a plain, non-degenerate
                                                   * negative control, not another landmine case). */
    enum { NBLK100 = CDIV(O100,128) * CDIV(I100,128) };  /* must be a compile-time constant expression for
                                                     * the static array below -- fp8_nblk() is a real
                                                     * function (runtime, not constexpr), so it can't
                                                     * size a `static` array even with literal inputs. */
    static uint8_t q100[O100*I100]; static float s100[NBLK100];
    for(int i=0;i<O100*I100;i++) q100[i]=rndbyte_nonan();
    for(int i=0;i<(int)(sizeof s100/sizeof *s100);i++) s100[i]=0.01f+0.001f*(float)i;

    static int8_t q1[O1*I1]; static float s1[O1];
    for(int i=0;i<O1*I1;i++) q1[i]=(int8_t)(rndbyte_nonan()-128);
    for(int i=0;i<O1;i++) s1[i]=0.02f+0.001f*(float)i;

    const char *dir="tests/tmp_fp8_snap";
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir,0755);
#endif
    char path[300]; snprintf(path,sizeof path,"%s/model.safetensors",dir);
    int64_t nb100=(int64_t)O100*I100, ns100=(int64_t)(sizeof s100);
    int64_t nb1=(int64_t)O1*I1, ns1=(int64_t)O1*4;
    char hdr[1024];
    int hl=snprintf(hdr,sizeof hdr,
        "{\"w100\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[0,%lld]},"
        "\"w100.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"w1\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"w1.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}}",
        (long long)nb100,(long long)nb100,
        (long long)(ns100/4),(long long)nb100,(long long)(nb100+ns100),
        (long long)nb1,(long long)(nb100+ns100),(long long)(nb100+ns100+nb1),
        (long long)O1,(long long)(nb100+ns100+nb1),(long long)(nb100+ns100+nb1+ns1));
    FILE *f=fopen(path,"wb");
    if(!f){ printf("FAIL: cannot create %s (run from c/, like tools/run_tests.py does)\n", path); fails++; return; }
    uint64_t hlen=(uint64_t)hl;
    fwrite(&hlen,8,1,f); fwrite(hdr,1,hl,f);
    fwrite(q100,1,(size_t)nb100,f); fwrite(s100,1,(size_t)ns100,f);
    fwrite(q1,1,(size_t)nb1,f); fwrite(s1,1,(size_t)ns1,f);
    fclose(f);

    static Model gm;                             /* only gm.S is used by qt_from_disk */
    st_init(&gm.S, dir);

    QT t100; memset(&t100,0,sizeof t100);
    qt_from_disk(&gm,"w100",O100,I100,8,0,&t100);
    CHECK(t100.fmt==100);
    CHECK(t100.q8!=NULL && t100.s!=NULL);            /* both weight and scale allocated (qalloc, not falloc) */
    static float dq_load[O100*I100], dq_ref[O100*I100];
    deq_fmt100(&t100,dq_load);
    QT tr100={.fmt=100,.q8=(int8_t*)q100,.s=s100,.O=O100,.I=I100};
    deq_fmt100(&tr100,dq_ref);
    CHECK(memcmp(dq_load,dq_ref,sizeof dq_ref)==0);

    QT t1; memset(&t1,0,sizeof t1);
    qt_from_disk(&gm,"w1",O1,I1,8,0,&t1);
    CHECK(t1.fmt==1);                            /* negative control: plain int8 still resolves as fmt=1 */
    static float dq_load1[O1*I1], dq_ref1[O1*I1];
    deq_fmt1(&t1,dq_load1);
    QT tr1={.fmt=1,.q8=q1,.s=s1,.O=O1,.I=I1};
    deq_fmt1(&tr1,dq_ref1);
    CHECK(memcmp(dq_load1,dq_ref1,sizeof dq_ref1)==0);

    unlink(path); rmdir(dir);
}

int main(void){
    test_disambiguation();
    test_loader_seam();
    if(fails){ printf("fp8 loader-seam tests: %d FAILED\n", fails); return 1; }
    printf("fp8 loader-seam tests: ok\n");
    return 0;
}

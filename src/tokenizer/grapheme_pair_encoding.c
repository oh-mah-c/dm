#include "tokenizer/grapheme_pair_encoding.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DM_GPU
#include "core/gpu/dm_gpu.h"

/* Simple intern table mapping symbol strings to uint32 IDs for the GPU path. */
typedef struct { char **names; uint32_t count, cap; } GpeIntern;

static void gpe_intern_free(GpeIntern *t) {
    for (uint32_t i = 0; i < t->count; i++) free(t->names[i]);
    free(t->names); t->names = NULL; t->count = t->cap = 0;
}

/* Returns ID (existing or new). Returns UINT32_MAX on allocation failure. */
static uint32_t gpe_intern_add(GpeIntern *t, const char *sym) {
    for (uint32_t i = 0; i < t->count; i++)
        if (strcmp(t->names[i], sym) == 0) return i;
    if (t->count == t->cap) {
        uint32_t nc = t->cap ? t->cap * 2 : 64;
        char **tmp = (char **)realloc(t->names, nc * sizeof(char *));
        if (!tmp) return UINT32_MAX;
        t->names = tmp; t->cap = nc;
    }
    t->names[t->count] = strdup(sym);
    if (!t->names[t->count]) return UINT32_MAX;
    return t->count++;
}

static uint32_t gpe_intern_lookup(const GpeIntern *t, const char *sym) {
    for (uint32_t i = 0; i < t->count; i++)
        if (strcmp(t->names[i], sym) == 0) return i;
    return UINT32_MAX;
}

#endif /* DM_GPU */

#define GPE_EOW "</w>"

typedef struct { char **items; size_t count, cap; } StrVec;
typedef struct { char *left, *right, *joined; } Merge;
typedef struct { Merge *items; size_t count, cap; } MergeTable;
typedef struct { char **syms; size_t len, cap, freq; } SymWord;
typedef struct { SymWord *items; size_t count, cap; } SymVocab;
typedef struct { char *text; size_t freq; } TextCount;
typedef struct { TextCount *items; size_t count, cap; } TextVocab;
typedef struct { char *left, *right; size_t freq; } PairStat;
typedef struct { PairStat *items; size_t count, cap; } PairStats;
typedef struct { char *unit; char *pretokenizer; MergeTable merges; } GPEModel;

static char *xstrdup(const char *s) { size_t n=strlen(s); char *o=(char*)malloc(n+1); if(!o) return NULL; memcpy(o,s,n+1); return o; }
static char *xstrndup(const char *s,size_t n){ char *o=(char*)malloc(n+1); if(!o) return NULL; memcpy(o,s,n); o[n]='\0'; return o; }
static char *concat2(const char *a,const char *b){ size_t na=strlen(a),nb=strlen(b); char *o=(char*)malloc(na+nb+1); if(!o) return NULL; memcpy(o,a,na); memcpy(o+na,b,nb+1); return o; }
static void json_string(FILE *out,const char *s){ fputc('"',out); for(;*s;s++){ unsigned char c=(unsigned char)*s; if(c=='"'||c=='\\'){fputc('\\',out);fputc(c,out);} else if(c=='\n')fputs("\\n",out); else if(c=='\r')fputs("\\r",out); else if(c=='\t')fputs("\\t",out); else if(c<32)fprintf(out,"\\u%04x",c); else fputc(c,out);} fputc('"',out); }

static void strvec_free(StrVec *v){ for(size_t i=0;i<v->count;i++) free(v->items[i]); free(v->items); v->items=NULL; v->count=v->cap=0; }
static int strvec_push_owned(StrVec *v,char *s){ if(v->count==v->cap){ size_t nc=v->cap?v->cap*2:16; char **t=(char**)realloc(v->items,nc*sizeof(char*)); if(!t)return -1; v->items=t; v->cap=nc;} v->items[v->count++]=s; return 0; }
static int strvec_push_copy(StrVec *v,const char *s){ char *c=xstrdup(s); if(!c)return -1; if(strvec_push_owned(v,c)!=0){free(c);return -1;} return 0; }

static size_t utf8_len(unsigned char c){ if((c&0x80u)==0)return 1; if((c&0xE0u)==0xC0u)return 2; if((c&0xF0u)==0xE0u)return 3; if((c&0xF8u)==0xF0u)return 4; return 1; }
static uint32_t utf8_cp(const char *s,size_t n){ const unsigned char *p=(const unsigned char*)s; if(n==1)return p[0]; if(n==2)return ((p[0]&0x1Fu)<<6)|(p[1]&0x3Fu); if(n==3)return ((p[0]&0x0Fu)<<12)|((p[1]&0x3Fu)<<6)|(p[2]&0x3Fu); if(n==4)return ((p[0]&0x07u)<<18)|((p[1]&0x3Fu)<<12)|((p[2]&0x3Fu)<<6)|(p[3]&0x3Fu); return p[0]; }
static int split_codepoints(const char *text,StrVec *out){ const unsigned char *p=(const unsigned char*)text; while(*p){ size_t n=utf8_len(*p); for(size_t i=1;i<n;i++) if((p[i]&0xC0u)!=0x80u){n=1;break;} char *c=xstrndup((const char*)p,n); if(!c||strvec_push_owned(out,c)!=0){free(c);return -1;} p+=n;} return 0; }
static int is_mark_cp(uint32_t cp){
    return (cp>=0x0300&&cp<=0x036F)||
           (cp>=0x0591&&cp<=0x05BD)||cp==0x05BF||(cp>=0x05C1&&cp<=0x05C2)||(cp>=0x05C4&&cp<=0x05C5)||
           (cp>=0x0610&&cp<=0x061A)||(cp>=0x064B&&cp<=0x065F)||(cp>=0x06D6&&cp<=0x06DC)||(cp>=0x06DF&&cp<=0x06E4)||(cp>=0x06E7&&cp<=0x06E8)||(cp>=0x06EA&&cp<=0x06ED)||
           (cp>=0x0900&&cp<=0x0903)||cp==0x093A||cp==0x093C||(cp>=0x0941&&cp<=0x0948)||cp==0x094D||(cp>=0x0951&&cp<=0x0957)||
           (cp>=0x0981&&cp<=0x0983)||cp==0x09BC||(cp>=0x09C1&&cp<=0x09C4)||cp==0x09CD||
           cp==0x0B82||(cp>=0x0BBE&&cp<=0x0BCD)||cp==0x0DCA;
}
static int is_virama_or_zwj(uint32_t cp){ return cp==0x094D||cp==0x0BCD||cp==0x0DCA||cp==0x200D; }

static int graphemes(const char *text,StrVec *out){ StrVec cps={0}; if(split_codepoints(text,&cps)!=0)return -1; char *cur=NULL; int join_next=0; for(size_t i=0;i<cps.count;i++){ uint32_t cp=utf8_cp(cps.items[i],strlen(cps.items[i])); if(!cur){ cur=xstrdup(cps.items[i]); if(!cur){strvec_free(&cps);return -1;} } else if(is_mark_cp(cp)||cp==0x200D||join_next){ char *n=concat2(cur,cps.items[i]); free(cur); cur=n; if(!cur){strvec_free(&cps);return -1;} } else { if(strvec_push_owned(out,cur)!=0){free(cur);strvec_free(&cps);return -1;} cur=xstrdup(cps.items[i]); if(!cur){strvec_free(&cps);return -1;} } if(is_virama_or_zwj(cp)) join_next=1; else if(!is_mark_cp(cp)) join_next=0; } if(cur&&strvec_push_owned(out,cur)!=0){free(cur);strvec_free(&cps);return -1;} strvec_free(&cps); return 0; }

static int atomic_units(const char *text,const char *unit,StrVec *out){ if(strcmp(unit,"grapheme")==0)return graphemes(text,out); if(strcmp(unit,"codepoint")==0)return split_codepoints(text,out); if(strcmp(unit,"byte")==0){ const unsigned char *p=(const unsigned char*)text; char buf[3]; while(*p){ snprintf(buf,sizeof(buf),"%02x",*p++); if(strvec_push_copy(out,buf)!=0)return -1;} return 0;} return -1; }
static size_t original_length(const char *text,const char *unit){ if(strcmp(unit,"byte")==0)return strlen(text); StrVec v={0}; if(atomic_units(text,unit,&v)!=0)return 0; size_t n=v.count; strvec_free(&v); return n; }

static int is_ascii_alpha(unsigned char c){ return (c>='A'&&c<='Z')||(c>='a'&&c<='z'); }
static int is_ascii_digit(unsigned char c){ return c>='0'&&c<='9'; }
static int pretokenize(const char *text,const char *mode,StrVec *out){
    if(strcmp(mode,"none")==0) return *text?strvec_push_copy(out,text):0;
    if(strcmp(mode,"whitespace")==0){ char *copy=xstrdup(text); if(!copy)return -1; char *s=copy; while(*s){ while(*s&&isspace((unsigned char)*s))s++; if(!*s)break; char *st=s; while(*s&&!isspace((unsigned char)*s))s++; char sv=*s; *s='\0'; if(strvec_push_copy(out,st)!=0){free(copy);return -1;} if(!sv)break; *s++=sv;} free(copy); return 0; }
    int gpt4=strcmp(mode,"gpt4")==0; if(strcmp(mode,"gpt2")!=0&&!gpt4)return -1;
    size_t i=0,n=strlen(text); while(i<n){ size_t st=i; if(text[i]=='\'' && i+1<n){ const char *tails[]={"s","t","re","ve","m","ll","d"}; for(size_t k=0;k<7;k++){ size_t l=strlen(tails[k]); if(i+1+l<=n&&strncmp(text+i+1,tails[k],l)==0){ if(strvec_push_owned(out,xstrndup(text+i,1+l))!=0)return -1; i+=1+l; goto cont; }} }
        if(text[i]==' '&&i+1<n&&!isspace((unsigned char)text[i+1])) i++;
        st=i;
        if(i<n&&is_ascii_alpha((unsigned char)text[i])){ while(i<n&&is_ascii_alpha((unsigned char)text[i]))i++; }
        else if(i<n&&is_ascii_digit((unsigned char)text[i])){ size_t max=gpt4?3:(size_t)-1; size_t c=0; while(i<n&&is_ascii_digit((unsigned char)text[i])&&c<max){i++;c++;} }
        else if(i<n&&!isspace((unsigned char)text[i])){ while(i<n&&!isspace((unsigned char)text[i])&&!is_ascii_alpha((unsigned char)text[i])&&!is_ascii_digit((unsigned char)text[i]))i++; }
        else { while(i<n&&isspace((unsigned char)text[i]))i++; }
        if(i>st){ size_t begin=(st>0&&text[st-1]==' '&&!isspace((unsigned char)text[st]))?st-1:st; if(strvec_push_owned(out,xstrndup(text+begin,i-begin))!=0)return -1; }
cont: ; }
    return 0;
}

static void text_vocab_free(TextVocab *v){ for(size_t i=0;i<v->count;i++)free(v->items[i].text); free(v->items); v->items=NULL; v->count=v->cap=0; }
static int text_vocab_add(TextVocab *v,const char *s,size_t f){ for(size_t i=0;i<v->count;i++) if(strcmp(v->items[i].text,s)==0){v->items[i].freq+=f;return 0;} if(v->count==v->cap){size_t nc=v->cap?v->cap*2:256; TextCount*t=(TextCount*)realloc(v->items,nc*sizeof(TextCount)); if(!t)return -1; v->items=t; v->cap=nc;} v->items[v->count].text=xstrdup(s); if(!v->items[v->count].text)return -1; v->items[v->count++].freq=f; return 0; }

static void symword_free(SymWord *w){ for(size_t i=0;i<w->len;i++)free(w->syms[i]); free(w->syms); memset(w,0,sizeof(*w)); }
static void symvocab_free(SymVocab *v){ for(size_t i=0;i<v->count;i++)symword_free(&v->items[i]); free(v->items); memset(v,0,sizeof(*v)); }
static int symword_from_token(SymWord *w,const char *tok,const char *unit,size_t freq){ StrVec u={0}; if(atomic_units(tok,unit,&u)!=0)return -1; if(strvec_push_copy(&u,GPE_EOW)!=0){strvec_free(&u);return -1;} w->syms=u.items; w->len=u.count; w->cap=u.cap; w->freq=freq; return 0; }
static int symvocab_from_text(const TextVocab *tv,const char *unit,SymVocab *sv){ sv->items=(SymWord*)calloc(tv->count?tv->count:1,sizeof(SymWord)); if(!sv->items)return -1; sv->cap=tv->count; for(size_t i=0;i<tv->count;i++){ if(symword_from_token(&sv->items[sv->count],tv->items[i].text,unit,tv->items[i].freq)!=0)return -1; sv->count++; } return 0; }

static void pairstats_free(PairStats *ps){ for(size_t i=0;i<ps->count;i++){free(ps->items[i].left);free(ps->items[i].right);} free(ps->items); memset(ps,0,sizeof(*ps)); }
static int pairstats_add(PairStats *ps,const char*l,const char*r,size_t f){ for(size_t i=0;i<ps->count;i++) if(strcmp(ps->items[i].left,l)==0&&strcmp(ps->items[i].right,r)==0){ps->items[i].freq+=f;return 0;} if(ps->count==ps->cap){size_t nc=ps->cap?ps->cap*2:256; PairStat*t=(PairStat*)realloc(ps->items,nc*sizeof(PairStat)); if(!t)return -1; ps->items=t; ps->cap=nc;} ps->items[ps->count].left=xstrdup(l); ps->items[ps->count].right=xstrdup(r); if(!ps->items[ps->count].left||!ps->items[ps->count].right)return -1; ps->items[ps->count++].freq=f; return 0; }
static int collect_pairs(const SymVocab *v,PairStats *ps){ for(size_t w=0;w<v->count;w++) for(size_t i=0;i+1<v->items[w].len;i++) if(pairstats_add(ps,v->items[w].syms[i],v->items[w].syms[i+1],v->items[w].freq)!=0)return -1; return 0; }
static const PairStat *best_pair(const PairStats *ps,size_t minf){ const PairStat *b=NULL; for(size_t i=0;i<ps->count;i++){ const PairStat*p=&ps->items[i]; if(!b||p->freq>b->freq||(p->freq==b->freq&&(strcmp(p->left,b->left)>0||(strcmp(p->left,b->left)==0&&strcmp(p->right,b->right)>0)))) b=p;} return (b&&b->freq>=minf)?b:NULL; }

static void merges_free(MergeTable *m){ for(size_t i=0;i<m->count;i++){free(m->items[i].left);free(m->items[i].right);free(m->items[i].joined);} free(m->items); memset(m,0,sizeof(*m)); }
static int merges_push(MergeTable*m,const char*l,const char*r){ if(m->count==m->cap){size_t nc=m->cap?m->cap*2:128; Merge*t=(Merge*)realloc(m->items,nc*sizeof(Merge)); if(!t)return -1; m->items=t; m->cap=nc;} Merge*x=&m->items[m->count]; x->left=xstrdup(l); x->right=xstrdup(r); x->joined=(x->left&&x->right)?concat2(l,r):NULL; if(!x->left||!x->right||!x->joined)return -1; m->count++; return 0; }
static int symword_merge(SymWord*w,const char*l,const char*r){ char **next=(char**)calloc(w->len?w->len:1,sizeof(char*)); if(!next)return -1; size_t o=0; for(size_t i=0;i<w->len;){ if(i+1<w->len&&strcmp(w->syms[i],l)==0&&strcmp(w->syms[i+1],r)==0){ next[o]=concat2(l,r); if(!next[o])goto bad; i+=2; o++; } else { next[o]=xstrdup(w->syms[i]); if(!next[o])goto bad; i++; o++; }} for(size_t i=0;i<w->len;i++)free(w->syms[i]); free(w->syms); w->syms=next; w->len=w->cap=o; return 0; bad: for(size_t i=0;i<o;i++)free(next[i]); free(next); return -1; }
static int symvocab_merge(SymVocab*v,const char*l,const char*r){ for(size_t i=0;i<v->count;i++) if(symword_merge(&v->items[i],l,r)!=0)return -1; return 0; }

static int read_training_tokens(char **paths,size_t n,const char *pretok,TextVocab *tv){ char line[32768]; for(size_t p=0;p<(n?n:1);p++){ FILE *fp=n?fopen(paths[p],"rb"):stdin; if(!fp){perror(paths[p]);return -1;} while(fgets(line,sizeof(line),fp)){ size_t ln=strlen(line); while(ln&&(line[ln-1]=='\n'||line[ln-1]=='\r'))line[--ln]='\0'; StrVec toks={0}; if(pretokenize(line,pretok,&toks)!=0){strvec_free(&toks); if(n)fclose(fp); return -1;} for(size_t i=0;i<toks.count;i++) if(text_vocab_add(tv,toks.items[i],1)!=0){strvec_free(&toks); if(n)fclose(fp); return -1;} strvec_free(&toks);} if(n)fclose(fp);} return 0; }
#ifdef DM_GPU
static int gpe_build_gpu_input(const SymVocab *sv, const GpeIntern *intern,
                                DmGpuBpeInput *out,
                                uint32_t **sid, uint32_t **wst,
                                uint32_t **wln, uint32_t **wfr) {
    size_t total = 0;
    for (size_t w = 0; w < sv->count; w++) total += sv->items[w].len;
    *sid = (uint32_t *)malloc(total * sizeof(uint32_t));
    *wst = (uint32_t *)malloc(sv->count * sizeof(uint32_t));
    *wln = (uint32_t *)malloc(sv->count * sizeof(uint32_t));
    *wfr = (uint32_t *)malloc(sv->count * sizeof(uint32_t));
    if (!*sid || !*wst || !*wln || !*wfr) { free(*sid); free(*wst); free(*wln); free(*wfr); return -1; }
    uint32_t off = 0;
    for (size_t w = 0; w < sv->count; w++) {
        (*wst)[w] = off; (*wln)[w] = (uint32_t)sv->items[w].len; (*wfr)[w] = (uint32_t)sv->items[w].freq;
        for (size_t s = 0; s < sv->items[w].len; s++) {
            uint32_t id = gpe_intern_lookup(intern, sv->items[w].syms[s]);
            if (id == UINT32_MAX) { free(*sid); free(*wst); free(*wln); free(*wfr); return -1; }
            (*sid)[off++] = id;
        }
    }
    out->sym_ids = *sid; out->total_syms = total;
    out->word_starts = *wst; out->word_lens = *wln;
    out->word_freqs = *wfr; out->n_words = sv->count;
    out->vocab_size = intern->count;
    return 0;
}
static int gpe_best_pair_gpu(const uint32_t *pair_counts, uint32_t vsz,
                              const GpeIntern *intern, size_t min_freq,
                              const char **lout, const char **rout) {
    uint32_t best = (uint32_t)min_freq, ba = UINT32_MAX, bb = UINT32_MAX;
    for (uint32_t a = 0; a < vsz; a++)
        for (uint32_t b = 0; b < vsz; b++) {
            uint32_t f = pair_counts[(size_t)a * vsz + b];
            if (f > best || (f == best && ba == UINT32_MAX)) { best = f; ba = a; bb = b; }
        }
    if (ba == UINT32_MAX) return 0;
    *lout = intern->names[ba]; *rout = intern->names[bb]; return 1;
}
#endif /* DM_GPU */

static int learn_gpe(char **paths, size_t n, const char *unit, const char *pretok,
                     size_t vocab_size, size_t minf, GPEModel *model, void *gpu_ctx) {
    TextVocab tv = {0};
    if (read_training_tokens(paths, n, pretok, &tv) != 0) return -1;
    SymVocab sv = {0};
    if (symvocab_from_text(&tv, unit, &sv) != 0) { text_vocab_free(&tv); return -1; }

    /* Count initial atomic symbols to determine how many merges to learn. */
    TextVocab init = {0};
    for (size_t w = 0; w < sv.count; w++)
        for (size_t s = 0; s < sv.items[w].len; s++)
            if (text_vocab_add(&init, sv.items[w].syms[s], 1) != 0) {
                symvocab_free(&sv); text_vocab_free(&tv); text_vocab_free(&init); return -1;
            }
    size_t target = vocab_size > init.count ? vocab_size - init.count : 0;

#ifdef DM_GPU
    DmGpuCtx *gpu = (DmGpuCtx *)gpu_ctx;
    GpeIntern intern = {0};
    if (gpu && dm_gpu_ready(gpu)) {
        /* Populate intern table from initial symbols. */
        for (size_t i = 0; i < init.count; i++)
            if (gpe_intern_add(&intern, init.items[i].text) == UINT32_MAX) {
                gpe_intern_free(&intern); gpu = NULL; /* fall back to CPU */
            }
    }
#else
    (void)gpu_ctx;
#endif
    text_vocab_free(&init);

    model->unit = xstrdup(unit); model->pretokenizer = xstrdup(pretok);
    if (!model->unit || !model->pretokenizer) {
#ifdef DM_GPU
        gpe_intern_free(&intern);
#endif
        symvocab_free(&sv); text_vocab_free(&tv); return -1;
    }

    for (size_t i = 0; i < target; i++) {
        char *l = NULL, *r = NULL; int found = 0;

#ifdef DM_GPU
        if (gpu && dm_gpu_ready(gpu) && intern.count <= DM_GPU_BPE_MAX_VOCAB) {
            uint32_t *sid = NULL, *wst = NULL, *wln = NULL, *wfr = NULL;
            DmGpuBpeInput inp = {0};
            if (gpe_build_gpu_input(&sv, &intern, &inp, &sid, &wst, &wln, &wfr) == 0) {
                size_t vsz = intern.count;
                uint32_t *pc = (uint32_t *)calloc(vsz * vsz, sizeof(uint32_t));
                if (pc && dm_gpu_bpe_pair_count(gpu, &inp, pc) == 0) {
                    const char *lp = NULL, *rp = NULL;
                    if (gpe_best_pair_gpu(pc, (uint32_t)vsz, &intern, minf, &lp, &rp)) {
                        l = xstrdup(lp); r = xstrdup(rp);
                        found = (l && r) ? 1 : 0;
                    } else { found = -1; }
                }
                free(pc); free(sid); free(wst); free(wln); free(wfr);
            }
        }
#endif

        if (!found) {
            PairStats ps = {0};
            if (collect_pairs(&sv, &ps) != 0) {
                pairstats_free(&ps); symvocab_free(&sv); text_vocab_free(&tv);
#ifdef DM_GPU
                gpe_intern_free(&intern);
#endif
                return -1;
            }
            const PairStat *b = best_pair(&ps, minf);
            if (!b) { pairstats_free(&ps); break; }
            l = xstrdup(b->left); r = xstrdup(b->right);
            pairstats_free(&ps);
            found = (l && r) ? 1 : 0;
        } else if (found < 0) { break; }

        if (!found || !l || !r ||
            merges_push(&model->merges, l, r) != 0 ||
            symvocab_merge(&sv, l, r) != 0) {
            free(l); free(r); symvocab_free(&sv); text_vocab_free(&tv);
#ifdef DM_GPU
            gpe_intern_free(&intern);
#endif
            return -1;
        }
#ifdef DM_GPU
        char *joined = concat2(l, r);
        if (joined) { gpe_intern_add(&intern, joined); free(joined); }
#endif
        free(l); free(r);
    }

#ifdef DM_GPU
    gpe_intern_free(&intern);
#endif
    symvocab_free(&sv); text_vocab_free(&tv); return 0;
}

static void strip_eow(StrVec *v){ if(!v->count)return; char *last=v->items[v->count-1]; size_t n=strlen(last),e=strlen(GPE_EOW); if(strcmp(last,GPE_EOW)==0){free(last);v->count--;} else if(n>=e&&strcmp(last+n-e,GPE_EOW)==0){last[n-e]='\0'; if(!*last){free(last);v->count--;}} }
static int encode_token(const GPEModel*m,const char*tok,StrVec*out){ if(atomic_units(tok,m->unit,out)!=0)return -1; if(strvec_push_copy(out,GPE_EOW)!=0)return -1; for(size_t mi=0;mi<m->merges.count;mi++){ StrVec next={0}; for(size_t i=0;i<out->count;){ Merge *mg=&m->merges.items[mi]; if(i+1<out->count&&strcmp(out->items[i],mg->left)==0&&strcmp(out->items[i+1],mg->right)==0){ char*j=concat2(out->items[i],out->items[i+1]); if(!j||strvec_push_owned(&next,j)!=0){free(j);strvec_free(&next);return -1;} i+=2; } else { if(strvec_push_copy(&next,out->items[i])!=0){strvec_free(&next);return -1;} i++; }} strvec_free(out); *out=next; if(out->count==1)break;} strip_eow(out); return 0; }
static int encode_line_model(const GPEModel*m,const char*line,StrVec*out){ StrVec toks={0}; if(pretokenize(line,m->pretokenizer,&toks)!=0)return -1; for(size_t i=0;i<toks.count;i++){ StrVec pcs={0}; if(encode_token(m,toks.items[i],&pcs)!=0){strvec_free(&pcs);strvec_free(&toks);return -1;} for(size_t j=0;j<pcs.count;j++){ char *owned=pcs.items[j]; if(strvec_push_owned(out,owned)!=0){pcs.items[j]=NULL;strvec_free(&pcs);strvec_free(&toks);return -1;} pcs.items[j]=NULL;} strvec_free(&pcs);} strvec_free(&toks); return 0; }

static int write_model(const GPEModel*m,const char*path){ FILE*out=fopen(path,"wb"); if(!out){perror(path);return -1;} fprintf(out,"{\n  \"version\": \"dm-gpe-coling-2025\",\n  \"unit\": "); json_string(out,m->unit); fprintf(out,",\n  \"pretokenizer\": "); json_string(out,m->pretokenizer); fprintf(out,",\n  \"merges\": [\n"); for(size_t i=0;i<m->merges.count;i++){ fprintf(out,"    ["); json_string(out,m->merges.items[i].left); fprintf(out,", "); json_string(out,m->merges.items[i].right); fprintf(out,"]%s\n",i+1<m->merges.count?",":""); } fprintf(out,"  ]\n}\n"); fclose(out); return 0; }
static char *slurp(const char*path){ FILE*fp=fopen(path,"rb"); if(!fp){perror(path);return NULL;} fseek(fp,0,SEEK_END); long n=ftell(fp); rewind(fp); char*b=(char*)malloc((size_t)n+1); if(!b){fclose(fp);return NULL;} size_t r=fread(b,1,(size_t)n,fp); b[r]='\0'; fclose(fp); return b; }
static char *json_get_string(const char*data,const char*key){ char pat[128]; snprintf(pat,sizeof(pat),"\"%s\"",key); char*p=strstr(data,pat); if(!p)return NULL; p=strchr(p,':'); if(!p)return NULL; p=strchr(p,'"'); if(!p)return NULL; p++; char*e=p; while(*e&&*e!='"')e++; return xstrndup(p,(size_t)(e-p)); }
static int read_model(const char*path,GPEModel*m){ char*d=slurp(path); if(!d)return -1; m->unit=json_get_string(d,"unit"); m->pretokenizer=json_get_string(d,"pretokenizer"); if(!m->unit||!m->pretokenizer){free(d);return -1;} char*p=strstr(d,"\"merges\""); if(p) p=strchr(p,'['); if(p) p++; while(p&&*p){ char *pair=strchr(p,'['); if(!pair)break; char *q=strchr(pair,'"'); if(!q)break; q++; char*e=strchr(q,'"'); if(!e)break; char*l=xstrndup(q,(size_t)(e-q)); q=strchr(e+1,'"'); if(!q){free(l);break;} q++; e=strchr(q,'"'); if(!e){free(l);break;} char*r=xstrndup(q,(size_t)(e-q)); if(!l||!r||merges_push(&m->merges,l,r)!=0){free(l);free(r);free(d);return -1;} free(l);free(r); p=e+1; } free(d); return 0; }
static void model_free(GPEModel*m){ free(m->unit); free(m->pretokenizer); merges_free(&m->merges); memset(m,0,sizeof(*m)); }

static int eval_model(const GPEModel*m,char**paths,size_t n,const char*len_unit,double*orig,double*toks){ char line[32768]; *orig=*toks=0; for(size_t p=0;p<n;p++){ FILE*fp=fopen(paths[p],"rb"); if(!fp){perror(paths[p]);return -1;} while(fgets(line,sizeof(line),fp)){ size_t ln=strlen(line); while(ln&&(line[ln-1]=='\n'||line[ln-1]=='\r'))line[--ln]='\0'; *orig+=(double)original_length(line,len_unit); StrVec out={0}; if(encode_line_model(m,line,&out)!=0){strvec_free(&out);fclose(fp);return -1;} *toks+=(double)out.count; strvec_free(&out);} fclose(fp);} return 0; }
static int pretoken_eval_paths(char**paths,size_t n,const char*pretok,const char*len_unit,double*orig,double*toks){ char line[32768]; *orig=*toks=0; for(size_t p=0;p<n;p++){ FILE*fp=fopen(paths[p],"rb"); if(!fp){perror(paths[p]);return -1;} while(fgets(line,sizeof(line),fp)){ size_t ln=strlen(line); while(ln&&(line[ln-1]=='\n'||line[ln-1]=='\r'))line[--ln]='\0'; *orig+=(double)original_length(line,len_unit); StrVec out={0}; if(pretokenize(line,pretok,&out)!=0){strvec_free(&out);fclose(fp);return -1;} *toks+=(double)out.count; strvec_free(&out);} fclose(fp);} return 0; }

static void usage(const char*prog){ fprintf(stderr,"Usage: %s gpe train -i <corpus...> -o model --unit grapheme|codepoint|byte --pretokenizer whitespace|gpt2|gpt4|none --vocab-size N\n",prog); fprintf(stderr,"       %s gpe encode -m model [-i input] [-o output] [--json-tokens]\n",prog); fprintf(stderr,"       %s gpe evaluate -m model -i <corpus...> [--length-unit unit]\n",prog); fprintf(stderr,"       %s gpe pretoken-eval -i <corpus...> [--reference <corpus...>] [--pretokenizer mode] [--length-unit unit]\n",prog); fprintf(stderr,"       %s gpe units text...\n",prog); }

int dm_gpe_cli(int argc,char**argv){ int start=1; if(argc>=2&&(strcmp(argv[1],"gpe")==0||strcmp(argv[1],"dm_gpe")==0))start=2; if(argc<=start){usage(argv[0]);return 2;} const char*cmd=argv[start];
    if(strcmp(cmd,"train")==0){ StrVec inputs={0}; const char*out=NULL,*unit="grapheme,*bad",*pretok="whitespace"; unit="grapheme"; size_t vs=0,minf=2; int stats=0,use_gpu=0,gpu_device=0; for(int i=start+1;i<argc;i++){ if((strcmp(argv[i],"-i")==0||strcmp(argv[i],"--input")==0)&&i+1<argc){ while(i+1<argc&&argv[i+1][0]!='-') if(strvec_push_copy(&inputs,argv[++i])!=0)return 1; } else if((strcmp(argv[i],"-o")==0||strcmp(argv[i],"--output")==0)&&i+1<argc)out=argv[++i]; else if(strcmp(argv[i],"--unit")==0&&i+1<argc)unit=argv[++i]; else if(strcmp(argv[i],"--pretokenizer")==0&&i+1<argc)pretok=argv[++i]; else if(strcmp(argv[i],"--vocab-size")==0&&i+1<argc)vs=(size_t)strtoull(argv[++i],NULL,10); else if(strcmp(argv[i],"--min-frequency")==0&&i+1<argc)minf=(size_t)strtoull(argv[++i],NULL,10); else if(strcmp(argv[i],"--stats")==0)stats=1; else if(strcmp(argv[i],"--gpu")==0)use_gpu=1; else if(strcmp(argv[i],"--gpu-device")==0&&i+1<argc){gpu_device=(int)strtol(argv[++i],NULL,10);use_gpu=1;} else {usage(argv[0]);strvec_free(&inputs);return 2;} } if(!out||!vs){usage(argv[0]);strvec_free(&inputs);return 2;} void*gpu_ctx=NULL;
#ifdef DM_GPU
        if(use_gpu){gpu_ctx=dm_gpu_create(gpu_device,NULL); if(!gpu_ctx||!dm_gpu_ready((DmGpuCtx*)gpu_ctx)){fprintf(stderr,"[gpe] GPU init failed, falling back to CPU\n");dm_gpu_destroy((DmGpuCtx*)gpu_ctx);gpu_ctx=NULL;} else { char _dname[256]={0}; dm_gpu_device_name((DmGpuCtx*)gpu_ctx,_dname,sizeof(_dname)); fprintf(stderr,"[gpe] GPU: %s\n",_dname); }}
#else
        if(use_gpu)fprintf(stderr,"[gpe] built without GPU support, using CPU\n");
#endif
        GPEModel m={0}; int rc=learn_gpe(inputs.items,inputs.count,unit,pretok,vs,minf,&m,gpu_ctx); if(rc==0)rc=write_model(&m,out); if(rc==0&&stats)fprintf(stderr,"{\"merges\": %zu, \"pretokenizer\": \"%s\", \"requested_vocab_size\": %zu, \"unit\": \"%s\"}\n",m.merges.count,m.pretokenizer,vs,m.unit); model_free(&m); strvec_free(&inputs);
#ifdef DM_GPU
        dm_gpu_destroy((DmGpuCtx*)gpu_ctx);
#endif
        return rc==0?0:1; }
    if(strcmp(cmd,"encode")==0){ const char*modelp=NULL,*input=NULL,*output=NULL; int json=0; for(int i=start+1;i<argc;i++){ if((strcmp(argv[i],"-m")==0||strcmp(argv[i],"--model")==0)&&i+1<argc)modelp=argv[++i]; else if((strcmp(argv[i],"-i")==0||strcmp(argv[i],"--input")==0)&&i+1<argc)input=argv[++i]; else if((strcmp(argv[i],"-o")==0||strcmp(argv[i],"--output")==0)&&i+1<argc)output=argv[++i]; else if(strcmp(argv[i],"--json-tokens")==0)json=1; else {usage(argv[0]);return 2;} } if(!modelp){usage(argv[0]);return 2;} GPEModel m={0}; if(read_model(modelp,&m)!=0)return 1; FILE*in=input?fopen(input,"rb"):stdin; FILE*out=output?fopen(output,"wb"):stdout; if(!in||!out){model_free(&m);return 1;} char line[32768]; while(fgets(line,sizeof(line),in)){ size_t ln=strlen(line); while(ln&&(line[ln-1]=='\n'||line[ln-1]=='\r'))line[--ln]='\0'; StrVec pcs={0}; if(encode_line_model(&m,line,&pcs)!=0){strvec_free(&pcs);model_free(&m);return 1;} if(json){ fputc('[',out); for(size_t i=0;i<pcs.count;i++){ if(i)fputs(", ",out); json_string(out,pcs.items[i]); } fputs("]\n",out);} else { for(size_t i=0;i<pcs.count;i++){ if(i)fputc(' ',out); fputs(pcs.items[i],out);} fputc('\n',out);} strvec_free(&pcs);} if(input)fclose(in); if(output)fclose(out); model_free(&m); return 0; }
    if(strcmp(cmd,"evaluate")==0){ const char*modelp=NULL,*lu="grapheme"; StrVec inputs={0}; for(int i=start+1;i<argc;i++){ if((strcmp(argv[i],"-m")==0||strcmp(argv[i],"--model")==0)&&i+1<argc)modelp=argv[++i]; else if((strcmp(argv[i],"-i")==0||strcmp(argv[i],"--input")==0)&&i+1<argc){ while(i+1<argc&&argv[i+1][0]!='-') if(strvec_push_copy(&inputs,argv[++i])!=0)return 1; } else if(strcmp(argv[i],"--length-unit")==0&&i+1<argc)lu=argv[++i]; else {usage(argv[0]);strvec_free(&inputs);return 2;} } GPEModel m={0}; if(!modelp||read_model(modelp,&m)!=0){strvec_free(&inputs);return 1;} double o,t; int rc=eval_model(&m,inputs.items,inputs.count,lu,&o,&t); if(rc==0)printf("{\n  \"compression_ratio\": %.12g,\n  \"original_length\": %.12g,\n  \"tokenized_length\": %.12g\n}\n",t?o/t:0.0,o,t); model_free(&m); strvec_free(&inputs); return rc==0?0:1; }
    if(strcmp(cmd,"pretoken-eval")==0){ StrVec in={0},ref={0}; const char*pretok="gpt2",*lu="grapheme"; int mode=0; for(int i=start+1;i<argc;i++){ if((strcmp(argv[i],"-i")==0||strcmp(argv[i],"--input")==0)&&i+1<argc){mode=1; while(i+1<argc&&argv[i+1][0]!='-') if(strvec_push_copy(&in,argv[++i])!=0)return 1;} else if(strcmp(argv[i],"--reference")==0&&i+1<argc){mode=2; while(i+1<argc&&argv[i+1][0]!='-') if(strvec_push_copy(&ref,argv[++i])!=0)return 1;} else if(strcmp(argv[i],"--pretokenizer")==0&&i+1<argc)pretok=argv[++i]; else if(strcmp(argv[i],"--length-unit")==0&&i+1<argc)lu=argv[++i]; else { (void)mode; usage(argv[0]);strvec_free(&in);strvec_free(&ref);return 2;} } double o,t,ro,rt; int rc=pretoken_eval_paths(in.items,in.count,pretok,lu,&o,&t); if(rc==0){ printf("{\n  \"cr_max\": %.12g,\n  \"original_length\": %.12g,\n  \"pretoken_count\": %.12g",t?o/t:0.0,o,t); if(ref.count&&pretoken_eval_paths(ref.items,ref.count,pretok,lu,&ro,&rt)==0) printf(",\n  \"tokenization_parity_to_reference\": %.12g",rt?t/rt:0.0); printf("\n}\n"); } strvec_free(&in);strvec_free(&ref); return rc==0?0:1; }
    if(strcmp(cmd,"units")==0){ for(int i=start+1;i<argc;i++){ StrVec b={0},c={0},g={0}; atomic_units(argv[i],"byte",&b); atomic_units(argv[i],"codepoint",&c); atomic_units(argv[i],"grapheme",&g); printf("{\"text\": "); json_string(stdout,argv[i]); printf(", \"bytes\": ["); for(size_t j=0;j<b.count;j++){if(j)printf(", ");json_string(stdout,b.items[j]);} printf("], \"codepoints\": ["); for(size_t j=0;j<c.count;j++){if(j)printf(", ");json_string(stdout,c.items[j]);} printf("], \"graphemes\": ["); for(size_t j=0;j<g.count;j++){if(j)printf(", ");json_string(stdout,g.items[j]);} printf("]}\n"); strvec_free(&b);strvec_free(&c);strvec_free(&g);} return 0; }
    usage(argv[0]); return 2; }

// Portable C++ implementation of the same maximal-reflection stereographic kernel.
// No fast-math: double precision, stable Student-t log CDF, exact coalescence.
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <algorithm>
namespace {
constexpr int D=100, P=101;
using Vec=std::array<double,P>;
double dot(const Vec&a,const Vec&b){double s=0;for(int i=0;i<P;++i)s+=a[i]*b[i];return s;}
void normalize(Vec&a){const double n=std::sqrt(dot(a,a));for(double&v:a)v/=n;}
double beta_cf(double a,double b,double x){
    constexpr double tiny=1e-300,eps=3e-14;
    const double qab=a+b,qap=a+1,qam=a-1;
    double c=1,d=1-qab*x/qap;if(std::abs(d)<tiny)d=tiny;
    d=1/d;double h=d;
    for(int m=1;m<=400;++m){
        const double m2=2.*m;
        double aa=m*(b-m)*x/((qam+m2)*(a+m2));
        d=1+aa*d;if(std::abs(d)<tiny)d=tiny;c=1+aa/c;if(std::abs(c)<tiny)c=tiny;
        d=1/d;h*=d*c;
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));
        d=1+aa*d;if(std::abs(d)<tiny)d=tiny;c=1+aa/c;if(std::abs(c)<tiny)c=tiny;
        d=1/d;const double delta=d*c;h*=delta;
        if(std::abs(delta-1)<eps)return h;
    }
    return std::numeric_limits<double>::quiet_NaN();
}
double logtcdf(double t){
    constexpr double a=51.,b=.5,df=102.;
    if(t==0)return -std::log(2.);
    static const double lb=std::lgamma(a)+std::lgamma(b)-std::lgamma(a+b);
    const double x=df/(df+t*t), y=t*t/(df+t*t);
    const double logbt=a*std::log(x)+b*std::log(y)-lb;
    double logib;
    if(x<(a+1)/(a+b+2))logib=logbt+std::log(beta_cf(a,b,x)/a);
    else{
        const double complement=std::exp(logbt)*beta_cf(b,a,y)/b;
        logib=std::log1p(-complement);
    }
    return t<0?-std::log(2.)+logib:std::log1p(-.5*std::exp(logib));
}
double logtarget(const Vec&z){
    const double gap=1-z[D],scale=10/gap;
    double n2=0;for(int i=0;i<D;++i)n2+=z[i]*z[i];n2*=scale*scale;
    const double argument=100*scale*(z[0]-z[1])*std::sqrt(102/(2+n2));
    return -51*std::log1p(n2/2)+logtcdf(argument)-100*std::log(gap);
}
double logq(const Vec&a,const Vec&p,double h){
    const double c=dot(a,p);
    if(c<=0)return -std::numeric_limits<double>::infinity();
    return -101*std::log(c)-(1/(c*c)-1)/(2*h*h);
}
int step(Vec&a,Vec&b,double&la,double&lb,const Vec&g,double w,double u,double h,int&aa,int&ab,int&common){
    Vec p,q,e;
    const double ag=dot(a,g);
    for(int i=0;i<P;++i)p[i]=a[i]+h*(g[i]-a[i]*ag);
    normalize(p);
    common=std::log(w)<=std::min(0.,logq(b,p,h)-logq(a,p,h));
    if(common)q=p;
    else{
        for(int i=0;i<P;++i)e[i]=a[i]-b[i];
        const double en=std::sqrt(dot(e,e));
        if(en>0)for(double&v:e)v/=en;
        const double ep=dot(e,p);
        for(int i=0;i<P;++i)q[i]=p[i]-2*e[i]*ep;
        normalize(q);
    }
    const double lp=logtarget(p),lq=common?lp:logtarget(q),lu=std::log(u);
    aa=lu<=std::min(0.,lp-la);ab=lu<=std::min(0.,lq-lb);
    if(aa){a=p;la=lp;}if(ab){b=q;lb=lq;}
    return a==b;
}
}
extern "C" double native_logtcdf(double t){return logtcdf(t);}
extern "C" double native_logtarget(const double*z){Vec a;std::copy(z,z+P,a.begin());return logtarget(a);}
extern "C" void native_step(const double*a0,const double*b0,const double*g0,double w,double u,double h,double*aout,double*bout,int*flags){
    Vec a,b,g;std::copy(a0,a0+P,a.begin());std::copy(b0,b0+P,b.begin());std::copy(g0,g0+P,g.begin());
    double la=logtarget(a),lb=logtarget(b);int aa,ab,common;
    flags[0]=step(a,b,la,lb,g,w,u,h,aa,ab,common);flags[1]=aa;flags[2]=ab;flags[3]=common;
    std::copy(a.begin(),a.end(),aout);std::copy(b.begin(),b.end(),bout);
}
extern "C" void native_run(const double*a0,const double*b0,std::uint64_t seed,int max_steps,double h,std::int64_t*out){
    Vec a,b,g;std::copy(a0,a0+P,a.begin());std::copy(b0,b0+P,b.begin());
    std::mt19937_64 rng(seed);std::normal_distribution<double> normal;
    auto uniform=[&](){return std::generate_canonical<double,53>(rng);};
    double la=logtarget(a),lb=logtarget(b);
    std::int64_t acca=0,accb=0,pm=0;int t=0,met=0;
    while(t<max_steps&&!met){
        for(double&v:g)v=normal(rng);
        const double w=uniform(),u=uniform();int aa,ab,common;
        met=step(a,b,la,lb,g,w,u,h,aa,ab,common);
        ++t;acca+=aa;accb+=ab;pm+=common;
    }
    out[0]=t;out[1]=met;out[2]=acca;out[3]=accb;out[4]=pm;
}

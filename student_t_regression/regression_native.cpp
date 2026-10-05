// Student-t regression: maximal-reflection coupling, no fast-math.
// X is row-major, standardized, with exactly 13 predictors (Boston Housing).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

namespace {
constexpr int D=14, P=15;
using Vec=std::array<double,P>;
constexpr double NEG_INF=-std::numeric_limits<double>::infinity();
double dot(const Vec&a,const Vec&b,int n){double s=0;for(int j=0;j<n;++j)s+=a[j]*b[j];return s;}
void normalize(Vec&v){double norm=std::sqrt(dot(v,v,P));for(double&x:v)x/=norm;}
struct Target {
    const double* X; const double* y; int n; double nu,R;
    double euclidean(const Vec&theta) const {
        double u=theta[D-1],sum=0,scale=std::exp(-2*u)/nu;
        for(int i=0;i<n;++i){
            double r=y[i];for(int j=0;j<D-1;++j)r-=X[i*(D-1)+j]*theta[j];
            double a=r*r*scale;
            if(std::isfinite(a)&&scale>0)sum+=std::log1p(a);
            else if(r!=0){double v=2*std::log(std::abs(r))-std::log(nu)-2*u;
                sum+=std::max(v,0.)+std::log1p(std::exp(-std::abs(v)));}
        }
        return -.5*(nu+1)*sum-n*u;
    }
    double operator()(const Vec&v,int method) const {
        if(method==0)return euclidean(v);
        double gap=1-v[D];if(!(gap>0))return NEG_INF;
        Vec theta{};for(int j=0;j<D;++j)theta[j]=R*v[j]/gap;
        return euclidean(theta)-D*std::log(gap);
    }
};
double logq(const Vec&a,const Vec&p,double h){
    double c=dot(a,p,P);if(c<=0)return NEG_INF;c=std::min(c,1.);
    return -P*std::log(c)-(1-c)*(1+c)/(2*h*h*c*c);
}
void proposals(int method,const Vec&a,const Vec&b,const Vec&g,double w,double h,Vec&p,Vec&q,bool&common){
    int dim=method?P:D;
    bool equal=std::equal(a.begin(),a.begin()+dim,b.begin());
    if(method){
        double ag=dot(a,g,P)/dot(a,a,P);
        for(int j=0;j<P;++j)p[j]=a[j]+h*(g[j]-a[j]*ag);
        normalize(p);
        common=equal||std::log(w)<=std::min(0.,logq(b,p,h)-logq(a,p,h));
        if(common){q=p;return;}
        Vec e{};for(int j=0;j<P;++j)e[j]=a[j]-b[j];
        double ep=dot(e,p,P)/dot(e,e,P);
        for(int j=0;j<P;++j)q[j]=p[j]-2*e[j]*ep;
        normalize(q);
    }else{
        Vec delta{};double ratio=0,norm2=0;
        for(int j=0;j<D;++j){p[j]=a[j]+h*g[j];delta[j]=(a[j]-b[j])/h;
            ratio-=g[j]*delta[j]+.5*delta[j]*delta[j];norm2+=delta[j]*delta[j];}
        common=equal||std::log(w)<=std::min(0.,ratio);
        if(common){q=p;return;}
        double eg=dot(delta,g,D)/norm2;
        for(int j=0;j<D;++j)q[j]=b[j]+h*(g[j]-2*delta[j]*eg);
    }
}
void step(const Target&t,int method,Vec&a,Vec&b,double&la,double&lb,const Vec&g,double w,double u,double h,int*flags){
    Vec p{},q{};bool common;proposals(method,a,b,g,w,h,p,q,common);
    double lp=t(p,method),lq=common?lp:t(q,method),lu=std::log(u);
    bool aa=lu<=std::min(0.,lp-la),ab=lu<=std::min(0.,lq-lb);
    if(std::isnan(lp)||lp==std::numeric_limits<double>::infinity())aa=false;
    if(std::isnan(lq)||lq==std::numeric_limits<double>::infinity())ab=false;
    if(aa){a=p;la=lp;}if(ab){b=q;lb=lq;}
    flags[0]=aa;flags[1]=ab;flags[2]=common;flags[3]=!std::isfinite(lp)+!std::isfinite(lq);
}
}
extern "C" double regression_logtarget(const double*X,const double*y,int n,double nu,double R,int method,const double*v){
    Vec a{};std::copy(v,v+(method?P:D),a.begin());return Target{X,y,n,nu,R}(a,method);
}
extern "C" void regression_step(const double*X,const double*y,int n,double nu,double R,int method,
    const double*a0,const double*b0,const double*g0,double w,double u,double h,double*aout,double*bout,int*flags){
    int dim=method?P:D;Vec a{},b{},g{};std::copy(a0,a0+dim,a.begin());std::copy(b0,b0+dim,b.begin());std::copy(g0,g0+dim,g.begin());
    Target t{X,y,n,nu,R};double la=t(a,method),lb=t(b,method);step(t,method,a,b,la,lb,g,w,u,h,flags);
    std::copy(a.begin(),a.begin()+dim,aout);std::copy(b.begin(),b.begin()+dim,bout);
}
extern "C" void regression_run(const double*X,const double*y,int n,double nu,double R,int method,
    const double*a0,const double*b0,std::uint64_t seed,int steps,double h,double tol,std::int64_t*out){
    int dim=method?P:D;Vec a{},b{},g{};std::copy(a0,a0+dim,a.begin());std::copy(b0,b0+dim,b.begin());
    Target target{X,y,n,nu,R};double la=target(a,method),lb=target(b,method);
    std::mt19937_64 rng(seed);std::normal_distribution<double> normal;
    auto uniform=[&](){return std::generate_canonical<double,53>(rng);};
    std::fill(out,out+8,0);
    for(int k=1;k<=steps;++k){
        for(int j=0;j<dim;++j)g[j]=normal(rng);
        double w=uniform(),u=uniform();int flags[4];step(target,method,a,b,la,lb,g,w,u,h,flags);
        out[1]+=flags[0];out[2]+=flags[1];out[6]+=flags[2];out[7]+=flags[3];
        if(out[0]==0){out[3]+=flags[0];out[4]+=flags[1];}
        bool equal=std::equal(a.begin(),a.begin()+dim,b.begin());
        if(out[5]==0&&equal)out[5]=k;
        if(out[0]==0){
            double dist2=0;
            for(int j=0;j<D;++j){double delta=method?R*(a[j]/(1-a[D])-b[j]/(1-b[D])):a[j]-b[j];dist2+=delta*delta;}
            if(equal||dist2<=tol*tol)out[0]=k;
        }
    }
}

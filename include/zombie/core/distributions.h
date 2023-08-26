#pragma once

#include <zombie/core/sampling.h>
#include "bessel.hpp"

namespace zombie {

template <int DIM>
class GreensFnFreeSpace {
public:
	// constructor
	GreensFnFreeSpace() {
		updatePole(Vector<DIM>::Zero());
	}

	// destructor
	virtual ~GreensFnFreeSpace() {}

	// updates the pole
	virtual void updatePole(const Vector<DIM>& x_, float rClamp_=1e-4f) {
		x = x_;
		rClamp = rClamp_;
	}

	// evaluates the Green's function
	virtual float evaluate(float r) const {
		return 0.0f;
	}

	// evaluates the gradient of the Green's function
	virtual Vector<DIM> gradient(float r, const Vector<DIM>& y) const {
		return Vector<DIM>::Zero();
	}

	// evaluates the Poisson Kernel (normal derivative of the Green's function)
	virtual float poissonKernel(float r, const Vector<DIM>& y, const Vector<DIM>& n) const {
		return 0.0f;
	}

	// evaluates the gradient of the Poisson Kernel
	virtual Vector<DIM> poissonKernelGradient(float r, const Vector<DIM>& y, const Vector<DIM>& n) const {
		return Vector<DIM>::Zero();
	}

	// evaluates the Green's function
	virtual float evaluate(const Vector<DIM>& y) const {
		float r = std::max(rClamp, (y - x).norm());
		return this->evaluate(r);
	}

	// evaluates the gradient of the Green's function
	virtual Vector<DIM> gradient(const Vector<DIM>& y) const {
		float r = std::max(rClamp, (y - x).norm());
		return this->gradient(r, y);
	}

	// evaluates the Poisson Kernel (normal derivative of the Green's function)
	virtual float poissonKernel(const Vector<DIM>& y, const Vector<DIM>& n) const {
		float r = std::max(rClamp, (y - x).norm());
		return this->poissonKernel(r, y, n);
	}

	// evaluates the gradient of the Poisson Kernel
	virtual Vector<DIM> poissonKernelGradient(const Vector<DIM>& y, const Vector<DIM>& n) const {
		float r = std::max(rClamp, (y - x).norm());
		return this->poissonKernelGradient(r, y, n);
	}

	// member
	Vector<DIM> x;
	float rClamp;
};

template <int DIM>
class HarmonicGreensFnFreeSpace: public GreensFnFreeSpace<DIM> {
public:
	// constructor
	HarmonicGreensFnFreeSpace(): GreensFnFreeSpace<DIM>() {
		std::cerr << "HarmonicGreensFnFreeSpac() not implemented for DIM: " << DIM << std::endl;
		exit(EXIT_FAILURE);
	}
};

template <>
class HarmonicGreensFnFreeSpace<2>: public GreensFnFreeSpace<2> {
public:
	// constructor
	HarmonicGreensFnFreeSpace(): GreensFnFreeSpace<2>() {}

	// evaluates the Green's function
	float evaluate(float r) const {
		return -std::log(r)/(2.0f*M_PI);
	}

	// evaluates the gradient of the Green's function
	Vector2 gradient(float r, const Vector2& y) const {
		Vector2 xy = x - y;
		float r2 = r*r;

		return -xy/(2.0f*M_PI*r2);
	}

	// evaluates the Poisson Kernel (normal derivative of the Green's function)
	float poissonKernel(float r, const Vector2& y, const Vector2& n) const {
		Vector2 xy = x - y;
		float r2 = r*r;

		return n.dot(xy)/(2.0f*M_PI*r2);
	}

	// evaluates the gradient of the Poisson Kernel
	Vector2 poissonKernelGradient(float r, const Vector2& y, const Vector2& n) const {
		Vector2 xy = x - y;
		float r2 = r*r;

		return (n - 2.0f*(n.dot(xy)/r2)*xy)/(2.0f*M_PI*r2);
	}
};

template <>
class HarmonicGreensFnFreeSpace<3>: public GreensFnFreeSpace<3> {
public:
	// constructor
	HarmonicGreensFnFreeSpace(): GreensFnFreeSpace<3>() {}

	// evaluates the Green's function
	float evaluate(float r) const {
		return 1.0f/(4.0f*M_PI*r);
	}

	// evaluates the gradient of the Green's function
	Vector3 gradient(float r, const Vector3& y) const {
		Vector3 xy = x - y;
		float r3 = r*r*r;

		return -xy/(4.0f*M_PI*r3);
	}

	// evaluates the Poisson Kernel (normal derivative of the Green's function)
	float poissonKernel(float r, const Vector3& y, const Vector3& n) const {
		Vector3 xy = x - y;
		float r3 = r*r*r;

		return n.dot(xy)/(4.0f*M_PI*r3);
	}

	// evaluates the gradient of the Poisson Kernel
	Vector3 poissonKernelGradient(float r, const Vector3& y, const Vector3& n) const {
		Vector3 xy = x - y;
		float r2 = r*r;
		float r3 = r*r*r;

		return (n - 3.0f*(n.dot(xy)/r2)*xy)/(4.0f*M_PI*r3);
	}
};

template <int DIM>
class YukawaGreensFnFreeSpace: public GreensFnFreeSpace<DIM> {
public:
	// constructor
	YukawaGreensFnFreeSpace(float lambda_): GreensFnFreeSpace<DIM>() {
		std::cerr << "YukawaGreensFnFreeSpace() not implemented for DIM: " << DIM << std::endl;
		exit(EXIT_FAILURE);
	}
};

template <>
class YukawaGreensFnFreeSpace<2>: public GreensFnFreeSpace<2> {
public:
	// constructor
	YukawaGreensFnFreeSpace(float lambda_):
		GreensFnFreeSpace<2>(), lambda(lambda_), sqrtLambda(std::sqrt(lambda_)) {}

	// evaluates the Green's function
	float evaluate(float r) const {
		float mur = r*sqrtLambda;
		float K0mur = bessel::bessk0(mur);

		return K0mur/(2.0*M_PI);
	}

	// evaluates the gradient of the Green's function
	Vector2 gradient(float r, const Vector2& y) const {
		Vector2 xy = x - y;
		float mur = r*sqrtLambda;
		float K1mur = bessel::bessk1(mur);
		float Qr = sqrtLambda*K1mur;

		return -xy*Qr/(2.0f*M_PI*r);
	}

	// evaluates the Poisson Kernel (normal derivative of the Green's function)
	float poissonKernel(float r, const Vector2& y, const Vector2& n) const {
		Vector2 xy = x - y;
		float mur = r*sqrtLambda;
		float K1mur = bessel::bessk1(mur);
		float Qr = sqrtLambda*K1mur;

		return n.dot(xy)*Qr/(2.0f*M_PI*r);
	}

	// evaluates the gradient of the Poisson Kernel
	Vector2 poissonKernelGradient(float r, const Vector2& y, const Vector2& n) const {
		Vector2 xy = x - y;
		float r2 = r*r;
		float mur = r*sqrtLambda;
		float K0mur = bessel::bessk0(mur);
		float K1mur = bessel::bessk1(mur);
		float K2mur = bessel::bessk(2, mur);
		float Qr1 = sqrtLambda*K1mur;
		float Qr2 = lambda*(K0mur + K2mur)/2.0f;

		return (n*Qr1 - (n.dot(xy)/r2)*(Qr1 + r*Qr2)*xy)/(2.0f*M_PI*r);
	}

	// members
	float lambda, sqrtLambda; // potential
};

template <>
class YukawaGreensFnFreeSpace<3>: public GreensFnFreeSpace<3> {
public:
	// constructor
	YukawaGreensFnFreeSpace(float lambda_):
		GreensFnFreeSpace<3>(), lambda(lambda_), sqrtLambda(std::sqrt(lambda_)) {}

	// evaluates the Green's function
	float evaluate(float r) const {
		float mur = r*sqrtLambda;
		float expmur = std::exp(-mur);

		return expmur/(4.0f*M_PI*r);
	}

	// evaluates the gradient of the Green's function
	Vector3 gradient(float r, const Vector3& y) const {
		Vector3 xy = x - y;
		float r2 = r*r;
		float mur = r*sqrtLambda;
		float expmur = std::exp(-mur);
		float Qr = sqrtLambda*expmur*(1.0f + 1.0f/mur);

		return -xy*Qr/(4.0f*M_PI*r2);
	}

	// evaluates the Poisson Kernel (normal derivative of the Green's function)
	float poissonKernel(float r, const Vector3& y, const Vector3& n) const {
		Vector3 xy = x - y;
		float r2 = r*r;
		float mur = r*sqrtLambda;
		float expmur = std::exp(-mur);
		float Qr = sqrtLambda*expmur*(1.0f + 1.0f/mur);

		return n.dot(xy)*Qr/(4.0f*M_PI*r2);
	}

	// evaluates the gradient of the Poisson Kernel
	Vector3 poissonKernelGradient(float r, const Vector3& y, const Vector3& n) const {
		Vector3 xy = x - y;
		float r2 = r*r;
		float mur = r*sqrtLambda;
		float expmur = std::exp(-mur);
		float Qr1 = sqrtLambda*expmur*(1.0f + 1.0f/mur);
		float Qr2 = sqrtLambda*expmur/mur;

		return (n*Qr1 - (n.dot(xy)/r2)*(2.0f*Qr1 + Qr2)*xy)/(4.0f*M_PI*r2);
	}

	// members
	float lambda, sqrtLambda; // potential
};

template <int DIM>
class GreensFnBall {
public:
	// constructor
	GreensFnBall() {
		updateBall(Vector<DIM>::Zero(), 0.0f);
	}

	// destructor
	virtual ~GreensFnBall() {}

	// updates the ball center and radius
	virtual void updateBall(const Vector<DIM>& c_, float R_, float rClamp_=1e-4f) {
		c = c_;
		yVol = Vector<DIM>::Zero();
		ySurf = Vector<DIM>::Zero();
		R = R_;
		r = 0.0f;
		rClamp = rClamp_;
	}

	// samples a point inside the ball given the direction along which to sample the point
	virtual Vector<DIM> sampleVolume(const Vector<DIM>& dir, pcg32& sampler, float& pdf) {
		return Vector<DIM>::Zero();
	}

	// samples a point inside the ball
	virtual Vector<DIM> sampleVolume(pcg32& sampler, float& pdf) {
		return Vector<DIM>::Zero();
	}

	// evaluates the Green's function
	virtual float evaluate() const {
		return 0.0f;
	}

	// evaluates the off-centered Green's function
	virtual float evaluate(const Vector<DIM>& x, const Vector<DIM>& y) const {
		return 0.0f;
	}

	// evaluates the gradient norm of the Green's function
	virtual float gradientNorm() const {
		return 0.0f;
	}

	// evaluates the gradient of the Green's function
	virtual Vector<DIM> gradient() const {
		return Vector<DIM>::Zero();
	}

	// evaluates the norm of the Green's function
	virtual float norm() const {
		return 0.0f;
	}

	// samples a point on the surface of the ball
	virtual Vector<DIM> sampleSurface(pcg32& sampler, float& pdf) {
		return Vector<DIM>::Zero();
	}

	// evaluates the Poisson Kernel (normal derivative of the Green's function)
	virtual float poissonKernel() const {
		return 0.0f;
	}

	// directly evaluates the centered Poisson Kernel at a point y over the
	// direction sampling pdf (i.e., HarmonicGreensFnFreeSpace::poissonKernel(y, n))
	virtual float directionSampledPoissonKernel(const Vector<DIM>& y) const {
		return 0.0f;
	}

	// evaluates the gradient of the Poisson Kernel
	virtual Vector<DIM> poissonKernelGradient() const {
		return Vector<DIM>::Zero();
	}

	// returns the probability of a random walking reaching the boundary of the ball
	virtual float potential() const {
		return 0.0f;
	}

	// members
	Vector<DIM> c, yVol, ySurf; // ball center, sampled point inside and on the surface of the ball
	float R, r; // ball radius and distance of the sampled point inside the ball to the ball center
	float rClamp;

protected:
	// samples a point inside the ball
	virtual Vector<DIM> rejectionSampleGreensFn(const Vector<DIM>& dir, float bound,
												pcg32& sampler, float& pdf) {
		int iter = 0;
		do {
			float u = sampler.nextFloat();
			r = sampler.nextFloat()*R;
			pdf = evaluate()/norm();
			float pdfRadius = pdf/pdfSampleSphereUniform<DIM>(r);
			iter++;

			if (u < pdfRadius/bound) {
				break;
			}

		} while (iter < 1000);

		r = std::max(rClamp, r);
		if (r > R) r = R/2.0f;
		yVol = c + r*dir;

		return yVol;
	}
};

template <int DIM>
class HarmonicGreensFnBall: public GreensFnBall<DIM> {
public:
	// constructor
	HarmonicGreensFnBall(): GreensFnBall<DIM>() {
		std::cerr << "HarmonicGreensFnBall() not implemented for DIM: " << DIM << std::endl;
		exit(EXIT_FAILURE);
	}
};

template <>
class HarmonicGreensFnBall<2>: public GreensFnBall<2> {
public:
	// constructor
	HarmonicGreensFnBall(): GreensFnBall<2>() {}

	// samples a point inside the ball given the direction along which to sample the point
	Vector2 sampleVolume(const Vector2& dir, pcg32& sampler, float& pdf) {
		// TODO: can probably do better
		// rejection sample radius r from pdf 4.0 * r * ln(R / r) / R^2
		float bound = 1.5f/R;

		return GreensFnBall<2>::rejectionSampleGreensFn(dir, bound, sampler, pdf);
	}

	// samples a point inside the ball
	Vector2 sampleVolume(pcg32& sampler, float& pdf) {
		return sampleVolume(sampleUnitSphereUniform<2>(sampler), sampler, pdf);
	}

	// evaluates the Green's function
	float evaluate() const {
		return std::log(R/r)/(2.0f*M_PI);
	}

	// evaluates the off-centered Green's function
	float evaluate(const Vector2& x, const Vector2& y) const {
		float r = std::max(rClamp, (y - x).norm());
		return (std::log(R*R - (x - c).dot(y - c)) - std::log(R*r))/(2.0f*M_PI);
	}

	// evaluates the gradient norm of the Green's function
	float gradientNorm() const {
		float r2 = r*r;
		return (1.0f/r2 - 1.0f/(R*R))/(2.0f*M_PI);
	}

	// evaluates the gradient of the Green's function
	Vector2 gradient() const {
		Vector2 d = yVol - c;
		return d*gradientNorm();
	}

	// evaluates the norm of the Green's function
	float norm() const {
		return R*R/4.0f;
	}

	// samples a point on the surface of the ball
	Vector2 sampleSurface(pcg32& sampler, float& pdf) {
		ySurf = c + R*sampleUnitSphereUniform<2>(sampler);
		pdf = 1.0f/(2.0f*M_PI);

		return ySurf;
	}

	// evaluates the Poisson Kernel (normal derivative of the Green's function)
	float poissonKernel() const {
		return 1.0f/(2.0f*M_PI);
	}

	// directly evaluates the centered Poisson Kernel at a point y over the
	// direction sampling pdf (i.e., HarmonicGreensFnFreeSpace::poissonKernel(y, n))
	float directionSampledPoissonKernel(const Vector2& y) const {
		return 1.0f;
	}

	// evaluates the gradient of the Poisson Kernel
	Vector2 poissonKernelGradient() const {
		Vector2 d = ySurf - c;

		return 2.0f*d/(2.0f*M_PI*R*R);
	}

	// returns the probability of a random walking reaching the boundary of the ball
	float potential() const {
		return 2.0f*M_PI*poissonKernel();
	}
};

template <>
class HarmonicGreensFnBall<3>: public GreensFnBall<3> {
public:
	// constructor
	HarmonicGreensFnBall(): GreensFnBall<3>() {}

	// samples a point inside the ball given the direction along which to sample the point
	Vector3 sampleVolume(const Vector3& dir, pcg32& sampler, float& pdf) {
		// sample radius r from pdf 6.0f * r * (R - r) / R^3 using Ulrich's polar method
		float u1 = sampler.nextFloat();
		float u2 = sampler.nextFloat();
		float phi = 2.0f*M_PI*u2;

		r = (1.0f + std::sqrt(1.0f - std::cbrt(u1*u1))*std::cos(phi))*R/2.0f;
		r = std::max(rClamp, r);
		if (r > R) r = R/2.0f;
		yVol = c + r*dir;
		pdf = evaluate()/norm();

		return yVol;
	}

	// samples a point inside the ball
	Vector3 sampleVolume(pcg32& sampler, float& pdf) {
		return sampleVolume(sampleUnitSphereUniform<3>(sampler), sampler, pdf);
	}

	// evaluates the Green's function
	float evaluate() const {
		return (1.0f/r - 1.0f/R)/(4.0f*M_PI);
	}

	// evaluates the off-centered Green's function
	float evaluate(const Vector3& x, const Vector3& y) const {
		float r = std::max(rClamp, (y - x).norm());
		return (1.0f/r - R/(R*R - (x - c).dot(y - c)))/(4.0f*M_PI);
	}

	// evaluates the gradient norm of the Green's function
	float gradientNorm() const {
		float r3 = r*r*r;
		return (1.0f/r3 - 1.0f/(R*R*R))/(4.0f*M_PI);
	}

	// evaluates the gradient of the Green's function
	Vector3 gradient() const {
		Vector3 d = yVol - c;
		return d*gradientNorm();
	}

	// evaluates the norm of the Green's function
	float norm() const {
		return R*R/6.0f;
	}

	// samples a point on the surface of the ball
	Vector3 sampleSurface(pcg32& sampler, float& pdf) {
		ySurf = c + R*sampleUnitSphereUniform<3>(sampler);
		pdf = 1.0f/(4.0f*M_PI);

		return ySurf;
	}

	// evaluates the Poisson Kernel (normal derivative of the Green's function)
	float poissonKernel() const {
		return 1.0f/(4.0f*M_PI);
	}

	// directly evaluates the centered Poisson Kernel at a point y over the
	// direction sampling pdf (i.e., HarmonicGreensFnFreeSpace::poissonKernel(y, n))
	float directionSampledPoissonKernel(const Vector3& y) const {
		return 1.0f;
	}

	// evaluates the gradient of the Poisson Kernel
	Vector3 poissonKernelGradient() const {
		Vector3 d = ySurf - c;

		return 3.0f*d/(4.0f*M_PI*R*R);
	}

	// returns the probability of a random walking reaching the boundary of the ball
	float potential() const {
		return 4.0f*M_PI*poissonKernel();
	}
};

template <int DIM>
class YukawaGreensFnBall: public GreensFnBall<DIM> {
public:
	// constructor
	YukawaGreensFnBall(float lambda_): GreensFnBall<DIM>() {
		std::cerr << "YukawaGreensFnBall() not implemented for DIM: " << DIM << std::endl;
		exit(EXIT_FAILURE);
	}
};

template <>
class YukawaGreensFnBall<2>: public GreensFnBall<2> {
public:
	// constructor
	YukawaGreensFnBall(float lambda_):
		GreensFnBall<2>(), lambda(lambda_), sqrtLambda(std::sqrt(lambda_)) {}

	// updates the ball center and radius
	void updateBall(const Vector2& c_, float R_, float rClamp_=1e-4f) {
		GreensFnBall<2>::updateBall(c_, R_, rClamp_);
		muR = R*sqrtLambda;
		K0muR = bessel::bessk0(muR);
		I0muR = bessel::bessi0(muR);
		K1muR = bessel::bessk1(muR);
		I1muR = bessel::bessi1(muR);
	}

	// samples a point inside the ball given the direction along which to sample the point
	Vector2 sampleVolume(const Vector2& dir, pcg32& sampler, float& pdf) {
		// TODO: can probably do better
		// rejection sample radius r from pdf r * λ * (K_0(r√λ) * I_0(R√λ) - I_0(r√λ) * K_0(R√λ)) / (I_0(R√λ) - 1)
		float bound = R <= lambda ?
					  std::max(std::max(2.2f/R, 2.2f/lambda), std::max(0.6f*std::sqrt(R), 0.6f*sqrtLambda)) :
					  std::max(std::min(2.2f/R, 2.2f/lambda), std::min(0.6f*std::sqrt(R), 0.6f*sqrtLambda));

		return GreensFnBall<2>::rejectionSampleGreensFn(dir, bound, sampler, pdf);
	}

	// samples a point inside the ball
	Vector2 sampleVolume(pcg32& sampler, float& pdf) {
		return sampleVolume(sampleUnitSphereUniform<2>(sampler), sampler, pdf);
	}

	// evaluates the Green's function
	float evaluate() const {
		float mur = r*sqrtLambda;
		float K0mur = bessel::bessk0(mur);
		float I0mur = bessel::bessi0(mur);

		return (K0mur - I0mur*K0muR/I0muR)/(2.0*M_PI);
	}

	// evaluates the off-centered Green's function
	float evaluate(const Vector2& x, const Vector2& y) const {
		// NOTE: this is an approximation of the infinite series expression,
		// except when x coincides with c
		float r1 = std::max(rClamp, (y - x).norm());
		float r2 = (R*R - (x - c).dot(y - c))/R;
		float mur1 = r1*sqrtLambda;
		float mur2 = r2*sqrtLambda;
		float K0mur1 = bessel::bessk0(mur1);
		float K0mur2 = bessel::bessk0(mur2);
		float I0mur1 = bessel::bessi0(mur1);
		float I0mur2 = bessel::bessi0(mur2);
		float Q1 = K0mur1 - I0mur1*K0muR/I0muR;
		float Q2 = K0mur2 - I0mur2*K0muR/I0muR;

		return (Q1 - Q2)/(2.0f*M_PI);
	}

	// evaluates the gradient norm of the Green's function
	float gradientNorm() const {
		float mur = r*sqrtLambda;
		float K1mur = bessel::bessk1(mur);
		float I1mur = bessel::bessi1(mur);
		float Qr = sqrtLambda*(K1mur - I1mur*K1muR/I1muR);

		return Qr/(2.0f*M_PI*r);
	}

	// evaluates the gradient of the Green's function
	Vector2 gradient() const {
		Vector2 d = yVol - c;
		return d*gradientNorm();
	}

	// evaluates the norm of the Green's function
	float norm() const {
		return (1.0f - 2.0*M_PI*poissonKernel())/lambda;
	}

	// samples a point on the surface of the ball
	Vector2 sampleSurface(pcg32& sampler, float& pdf) {
		ySurf = c + R*sampleUnitSphereUniform<2>(sampler);
		pdf = 1.0f/(2.0f*M_PI);

		return ySurf;
	}

	// evaluates the Poisson Kernel (normal derivative of the Green's function)
	float poissonKernel() const {
		return 1.0f/(2.0f*M_PI*I0muR);
	}

	// directly evaluates the centered Poisson Kernel at a point y over the
	// direction sampling pdf (i.e., HarmonicGreensFnFreeSpace::poissonKernel(y, n))
	float directionSampledPoissonKernel(const Vector2& y) const {
		float r = std::max(rClamp, (y - c).norm());
		float mur = r*sqrtLambda;
		float K1mur = bessel::bessk1(mur);
		float I1mur = bessel::bessi1(mur);
		float Q = K1mur + I1mur*K0muR/I0muR;

		return mur*Q;
	}

	// evaluates the gradient of the Poisson Kernel
	Vector2 poissonKernelGradient() const {
		Vector2 d = ySurf - c;
		float QR = sqrtLambda/(R*I1muR);

		return d*QR/(2.0f*M_PI);
	}

	// returns the probability of a random walking reaching the boundary of the ball
	float potential() const {
		return 2.0f*M_PI*poissonKernel();
	}

private:
	// members
	float lambda, sqrtLambda; // potential
	float muR, K0muR, I0muR, K1muR, I1muR;
};

template <>
class YukawaGreensFnBall<3>: public GreensFnBall<3> {
public:
	// constructor
	YukawaGreensFnBall(float lambda_):
		GreensFnBall<3>(), lambda(lambda_), sqrtLambda(std::sqrt(lambda_)) {}

	// updates the ball center and radius
	void updateBall(const Vector3& c_, float R_, float rClamp_=1e-4f) {
		GreensFnBall<3>::updateBall(c_, R_, rClamp_);
		muR = R*sqrtLambda;
		expmuR = std::exp(-muR);
		float exp2muR = expmuR*expmuR;
		float coshmuR = (1.0f + exp2muR)/(2.0f*expmuR);
		sinhmuR = (1.0f - exp2muR)/(2.0f*expmuR);
		K32muR = expmuR*(1.0f + 1.0f/muR);
		I32muR = coshmuR - sinhmuR/muR;
	}

	// samples a point inside the ball given the direction along which to sample the point
	Vector3 sampleVolume(const Vector3& dir, pcg32& sampler, float& pdf) {
		// TODO: can probably do better
		// rejection sample radius r from pdf r * λ * sinh((R - r)√λ) / (sinh(R√λ) - R√λ)
		float bound = R <= lambda ?
					  std::max(std::max(2.0f/R, 2.0f/lambda), std::max(0.5f*std::sqrt(R), 0.5f*sqrtLambda)) :
					  std::max(std::min(2.0f/R, 2.0f/lambda), std::min(0.5f*std::sqrt(R), 0.5f*sqrtLambda));

		return GreensFnBall<3>::rejectionSampleGreensFn(dir, bound, sampler, pdf);
	}

	// samples a point inside the ball
	Vector3 sampleVolume(pcg32& sampler, float& pdf) {
		return sampleVolume(sampleUnitSphereUniform<3>(sampler), sampler, pdf);
	}

	// evaluates the Green's function
	float evaluate() const {
		float mur = r*sqrtLambda;
		float expmur = std::exp(-mur);
		float sinhmur = (1.0f - expmur*expmur)/(2.0f*expmur);

		return (expmur - expmuR*sinhmur/sinhmuR)/(4.0f*M_PI*r);
	}

	// evaluates the off-centered Green's function
	float evaluate(const Vector3& x, const Vector3& y) const {
		// NOTE: this is an approximation of the infinite series expression,
		// except when x coincides with c
		float r1 = std::max(rClamp, (y - x).norm());
		float r2 = (R*R - (x - c).dot(y - c))/R;
		float mur1 = r1*sqrtLambda;
		float mur2 = r2*sqrtLambda;
		float expmur1 = std::exp(-mur1);
		float expmur2 = std::exp(-mur2);
		float sinhmur1 = (1.0f - expmur1*expmur1)/(2.0f*expmur1);
		float sinhmur2 = (1.0f - expmur2*expmur2)/(2.0f*expmur2);
		float Q1 = (expmur1 - expmuR*sinhmur1/sinhmuR)/r1;
		float Q2 = (expmur2 - expmuR*sinhmur2/sinhmuR)/r2;

		return (Q1 - Q2)/(4.0f*M_PI);
	}

	// evaluates the gradient norm of the Green's function
	float gradientNorm() const {
		float r2 = r*r;
		float mur = r*sqrtLambda;
		float expmur = std::exp(-mur);
		float exp2mur = expmur*expmur;
		float coshmur = (1.0f + exp2mur)/(2.0f*expmur);
		float sinhmur = (1.0f - exp2mur)/(2.0f*expmur);
		float K32mur = expmur*(1.0f + 1.0f/mur);
		float I32mur = coshmur - sinhmur/mur;
		float Qr = sqrtLambda*(K32mur - I32mur*K32muR/I32muR);

		return Qr/(4.0f*M_PI*r2);
	}

	// evaluates the gradient of the Green's function
	Vector3 gradient() const {
		Vector3 d = yVol - c;
		return d*gradientNorm();
	}

	// evaluates the norm of the Green's function
	float norm() const {
		return (1.0f - 4.0*M_PI*poissonKernel())/lambda;
	}

	// samples a point on the surface of the ball
	Vector3 sampleSurface(pcg32& sampler, float& pdf) {
		ySurf = c + R*sampleUnitSphereUniform<3>(sampler);
		pdf = 1.0f/(4.0f*M_PI);

		return ySurf;
	}

	// evaluates the Poisson Kernel (normal derivative of the Green's function)
	float poissonKernel() const {
		return muR/(4.0f*M_PI*sinhmuR);
	}

	// directly evaluates the centered Poisson Kernel at a point y over the
	// direction sampling pdf (i.e., HarmonicGreensFnFreeSpace::poissonKernel(y, n))
	float directionSampledPoissonKernel(const Vector3& y) const {
		float r = std::max(rClamp, (y - c).norm());
		float mur = r*sqrtLambda;
		float expmur = std::exp(-mur);
		float exp2mur = expmur*expmur;
		float coshmur = (1.0f + exp2mur)/(2.0f*expmur);
		float sinhmur = (1.0f - exp2mur)/(2.0f*expmur);
		float K32mur = expmur*(1.0f + 1.0f/mur);
		float I32mur = coshmur - sinhmur/mur;
		float Q = K32mur + I32mur*expmuR/sinhmuR;

		return mur*Q;
	}

	// evaluates the gradient of the Poisson Kernel
	Vector3 poissonKernelGradient() const {
		Vector3 d = ySurf - c;
		float QR = lambda/I32muR;

		return d*QR/(4.0f*M_PI);
	}

	// returns the probability of a random walking reaching the boundary of the ball
	float potential() const {
		return 4.0f*M_PI*poissonKernel();
	}

private:
	// members
	float lambda, sqrtLambda; // potential
	float muR, expmuR, sinhmuR, K32muR, I32muR;
};

} // zombie

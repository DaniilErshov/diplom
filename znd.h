#include "consts.h"

//double phi;

void integrate(double *Y1, double *Y2, double *Pressure, double *Density, double *Velocity, double *Coordinate, double *M, double *Thermicity, double *SoundSpeed, double *dt);
int init_jacobian( int* n, double* t, double* y, int* ml, int* mu, double* dypdy, int* nrowpd );
int init_right_part_H2_multiple_reaction_22(int *n, double* t, double* y, double* yprime);
void get_vN_parameters(double *w, double *rho, double *P, double *Y);
void set_initial_parameters(double *Y, double *Velocity, double *Density, double *Pressure, double *Temperature);
void znd_structure();
void CJ_velocity(double *Y, double *Velocity, double *Density, double *Pressure, double *Temperature);
void CJ_equilibrate(double *Y1, double *Y2, double *Density, double *Temperature, double *dt);
int init_right_part_H2_multiple_reaction_22_CJ(int *n, double* t, double* y, double* yprime);

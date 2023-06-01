#include "dlsode.h"
#include "znd.h"
#include "consts.h"
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;

// Number of chemical reactions
static int num_react;
// Number of species
static int num_gas_species;

// Square of frozen sound speed
double af2;
// Square of Mach Number
double M2;
// Thermicity
double sigma;
// Gamma = Cp / Cv
double gamma;
double phi = 0.6;

extern "C" {
    EXT_FORTRAN_FUNC(int, dlsode, int (*f)(int*, double*, double*, double*),
        int* neq, double* y, double* t, double* tout, int* irol, double* rtol,
        double* atoll, int* itask, int* istate, int* iopt, double* rwork, int* lrw,
        int* iwork, int* liw, int (*jac)(int*, double*, double*, int*, int*, double*, int*), int* mf);
}

int init_right_part_H2_multiple_reaction_22(int *n, double* t, double* y, double* yprime) {
// chemistry using a 22 reversible reactions kinetics for H2 burning from
// Alan Keromnes et al. An experimental and detailed chemical kinetic modeling study of hydrogen and syngas mixture oxidation at elevated pressures // Combustion and Flame 160 (2013) 995–1011
    
    double k_0_f[3], k_inf_f[3], k_0_r[3], k_inf_r[3];
    double c[3], m[3], d = 0.14;
    double Pr_f[3], Pr_r[3];
    int k = 0, l = 0;
    double logF_f, logF_core_f, logF_r, logF_core_r;

    double sum1, sum2;

    sum1 = 0.0;
    for(int i = 0; i < num_gas_species; i++)
        sum1 += y[i];

    double T1 = y[num_gas_species] / phyc.kR / sum1;
    cout << "T1 = " << T1 << endl;
    
    double* forward = new double[num_react];
    double* reverse = new double[num_react];
    double* equilib = new double[num_react];

    for (int i = 0; i < num_react; i++) {
        if (i != 8 && i!= 15 && i!= 16) {
            forward[i] = chec.kPrex_f[i] * pow(T1, chec.kPow_f[i])
                * exp(-chec.kE_f[i] / T1 / phyc.kRc);
            reverse[i] = chec.kPrex_r[i] * pow(T1, chec.kPow_r[i])
                * exp(-chec.kE_r[i] / T1 / phyc.kRc);
        }
        else {
            if (i == 8) k = 0;
            if (i == 15) k = 1;
            if (i == 16) k = 2;

            k_inf_f[k] = chec.kPrex_f[i] * pow(T1, chec.kPow_f[i])
                * exp(-chec.kE_f[i] / T1 / phyc.kRc);
            k_inf_r[k] = chec.kPrex_r[i] * pow(T1, chec.kPow_r[i])
                * exp(-chec.kE_r[i] / T1 / phyc.kRc);
            k_0_f[k] = chec.kPrex_f_lp[i] * pow(T1, chec.kPow_f_lp[i])
                * exp(-chec.kE_f_lp[i] / T1 / phyc.kRc);
            k_0_r[k] = chec.kPrex_r_lp[i] * pow(T1, chec.kPow_r_lp[i])
                * exp(-chec.kE_r_lp[i] / T1 / phyc.kRc);

            c[k] = -0.4 - 0.67 * log10(chec.Fcent[i]);
            m[k] = 0.75 - 1.27 * log10(chec.Fcent[i]);
        }
    }
        
    Pr_f[0] = (k_0_f[0] * (1.3 * y[0] + 10 * y[6] + y[1] + y[2] + y[3] + y[4] + y[5] + y[7] + y[8])) / k_inf_f[0];
    Pr_r[0] = (k_0_r[0] * (1.3 * y[0] + 10 * y[6] + y[1] + y[2] + y[3] + y[4] + y[5] + y[7] + y[8])) / k_inf_r[0];
    Pr_f[1] = (k_0_f[1] * (3.7 * y[0] + 1.5 * y[8] + 1.2 * y[2] + 7.7 * y[7] + y[1] + y[3] + y[4] + y[5] + y[6])) / k_inf_f[1];
    Pr_r[1] = (k_0_r[1] * (3.7 * y[0] + 1.5 * y[8] + 1.2 * y[2] + 7.7 * y[7] + y[1] + y[3] + y[4] + y[5] + y[6])) / k_inf_r[1];
    Pr_f[2] = (k_0_f[2] * y[6]) / k_inf_f[2];
    Pr_r[2] = (k_0_r[2] * y[6]) / k_inf_r[2];

    for (k = 0; k < 3; k++) {
        if (k == 0) l = 8;
        if (k == 1) l = 15;
        if (k == 2) l = 16;

        if (Pr_f[k] == 0) forward[l] = k_inf_f[k];
        else {
            logF_core_f = pow((log10(Pr_f[k]) + c[k]) / (m[k] - d * (log10(Pr_f[k]) + c[k])), 2);
            logF_f = pow(1.0 + logF_core_f,-1) * log10(chec.Fcent[l]);
            chec.F_f[l] = pow(10, logF_f);
            forward[l] = k_inf_f[k] * (Pr_f[k] / (1 + Pr_f[k])) * chec.F_f[l];
        }

        if (Pr_r[k] == 0) reverse[l] = k_inf_r[k];
        else {
            logF_core_r = pow((log10(Pr_r[k]) + c[k]) / (m[k] - d * (log10(Pr_r[k]) + c[k])), 2);
            logF_r = pow(1.0 + logF_core_r,-1) * log10(chec.Fcent[l]);
            chec.F_r[l] = pow(10, logF_r);
            reverse[l] = k_inf_r[k] * (Pr_r[k] / (1 + Pr_r[k])) * chec.F_r[l];
        }
    }

    equilib[0] = forward[0] * y[1] * y[2] - reverse[0] * y[3] * y[4];
    equilib[1] = forward[1] * y[0] * y[3] - reverse[1] * y[1] * y[4];
    equilib[2] = forward[2] * y[0] * y[4] - reverse[2] * y[1] * y[6];
    equilib[3] = forward[3] * y[6] * y[3] - reverse[3] * y[4] * y[4];
    equilib[4] = (forward[4] * y[0] - reverse[4] * y[1] * y[1]) *
        (2.5 * y[0] + 12 * y[6] + y[1] + y[2] + y[3] + y[4] + y[5] + y[7] + y[8]);
    equilib[5] = (forward[5] * y[3] * y[3] - reverse[5] * y[2]) *
        (2.5 * y[0] + 12 * y[6] + y[1] + y[2] + y[3] + y[4] + y[5] + y[7] + y[8]);
    equilib[6] = (forward[6] * y[3] * y[1] - reverse[6] * y[4]) *
        (2.5 * y[0] + 12 * y[6] + y[1] + y[2] + y[3] + y[4] + y[5] + y[7] + y[8]);
    equilib[7] = (forward[7] * y[1] * y[4] - reverse[7] * y[6]) *
        (0.73 * y[0] + 3.65 * y[6] + y[1] + y[2] + y[3] + y[4] + y[5] + y[7] + y[8]);
    equilib[8] = forward[8] * y[1] * y[2] - reverse[8] * y[5];
    equilib[9] = forward[9] * y[0] * y[2] - reverse[9] * y[1] * y[5];
    equilib[10] = forward[10] * y[5] * y[1] - reverse[10] * y[4] * y[4];
    equilib[11] = forward[11] * y[5] * y[3] - reverse[11] * y[4] * y[2];
    equilib[12] = forward[12] * y[5] * y[4] - reverse[12] * y[6] * y[2];
    equilib[13] = forward[13] * y[5] * y[5] - reverse[13] * y[7] * y[2]; 
    equilib[14] = forward[14] * y[5] * y[5] - reverse[14] * y[7] * y[2]; 
    equilib[15] = forward[15] * y[7] - reverse[15] * y[4] * y[4];
    equilib[16] = forward[16] * y[7] - reverse[16] * y[4] * y[4]; 
    equilib[17] = forward[17] * y[7] * y[1] - reverse[17] * y[6] * y[4];  
    equilib[18] = forward[18] * y[7] * y[1] - reverse[18] * y[5] * y[0];
    equilib[19] = forward[19] * y[7] * y[3] - reverse[19] * y[4] * y[5];
    equilib[20] = forward[20] * y[7] * y[4] - reverse[20] * y[5] * y[6];
    equilib[21] = forward[21] * y[7] * y[4] - reverse[21] * y[5] * y[6];

    yprime[0] = -equilib[1] - equilib[2] - equilib[4] - equilib[9] + equilib[18];
    yprime[1] = -equilib[0] + equilib[4] - equilib[6] - equilib[7] - equilib[8] - 
        equilib[10] - equilib[17] - yprime[0];
    yprime[2] = -equilib[0] + equilib[5] - equilib[8]  - equilib[9] + equilib[11] + 
        equilib[12] + equilib[13] + equilib[14];
    yprime[3] = equilib[0] - equilib[1] - equilib[3] - 2. * equilib[5] - equilib[6] - 
        equilib[11] - equilib[19];
    yprime[4] = equilib[0] + equilib[1] - equilib[2] + 2. * equilib[3] + equilib[6] - 
        equilib[7] + 2. * equilib[10] + equilib[11] - equilib[12]  + 2. * equilib[15] + 
        2. * equilib[16] + equilib[17] + equilib[19] - equilib[20] - equilib[21];
    yprime[5] = equilib[8] + equilib[9] - equilib[10] - equilib[11]  - equilib[12] - 
        2. * equilib[13] - 2. * equilib[14] + equilib[18] + equilib[19] + equilib[20] + 
        equilib[21];
    yprime[6] = equilib[2] - equilib[3] + equilib[7] + equilib[12] + equilib[17] + 
        equilib[20] + equilib[21];
    yprime[7] = equilib[13] + equilib[14] - equilib[15] - equilib[16] - equilib[17] - 
        equilib[18] - equilib[19] - equilib[20] - equilib[21];
    yprime[8] = 0;
    
    double sigma_i;
    // Molar weight of mixture
    double W;
    // Cp
    double Cp;
    // Cv
    double Cv;
    // eta = 1 - M^2, M = w / af2 -- Mach number
    double eta;
    // Mass fraction
    double Yi, Yiprime;

    sum1 = 0.0;
    sum2 = 0.0;
    for(int i = 0; i < num_gas_species; i++)
    {
        Yi = y[i] * phyc.mol_weight[i] / y[num_gas_species + 1];
        sum1 += y[i];
        sum2 += Yi * get_Cpi(i, T1);
    }

    W = y[num_gas_species + 1] / sum1;
    //cout << "W = " << W << endl;
    Cp = sum2;
    Cv = Cp - phyc.kR / W;
    gamma = Cp / Cv;
    //cout << "gamma = " << gamma << endl;
    af2 = gamma * y[num_gas_species] / y[num_gas_species + 1];
    M2 = pow(y[num_gas_species + 2], 2) / af2;
    eta = 1 - M2;

    sum1 = 0.0;
    for(int i = 0; i < num_gas_species; i++)
    {
        sigma_i = (W / phyc.mol_weight[i]) - (get_Hi(i, T1) / Cp / T1);
        Yiprime = phyc.mol_weight[i] * yprime[i] / y[num_gas_species + 1];
        sum1 += sigma_i * Yiprime;
    }
    sigma = sum1;
    //cout << "sigma = " << sigma << endl;

    //cout << "af2 = " << af2 << ", eta = " << eta << ", sigma = " << sigma << endl;

    // dP/dt
    yprime[num_gas_species] = -1.0 * y[num_gas_species + 1] * pow(y[num_gas_species + 2], 2) * sigma / eta;
    // drho/dt
    yprime[num_gas_species + 1] = -1.0 * y[num_gas_species + 1] * sigma / eta;
    // dw/dt
    yprime[num_gas_species + 2] = y[num_gas_species + 2] * sigma / eta;
    // dx/dt
    yprime[num_gas_species + 3] = y[num_gas_species + 2];

    delete [] forward;
    delete [] reverse;
    delete [] equilib;

    return 0;
}


int init_jacobian( int* n, double* t, double* y, int* ml, int* mu, double* dypdy, int* nrowpd ) {
    return 0;
}

void integrate(double *Y1, double *Y2, double *Pressure, double *Density, double *Velocity, double *Coordinate, double *M, double *Thermicity, double *SoundSpeed, double *dt)
{
    int NEQ, ITOL, ITASK, ISTATE, IOPT, LRW, LIW, MF;

    NEQ = num_gas_species + 4; // Number of first-order ODE's
    
    MF = 22; // Method flag
    LRW = 22 +  9*NEQ + NEQ*NEQ; // for MF = 21 or 22
    LIW = 20 + NEQ; // for MF = 21 or 22

    ITASK = 1;
    ITOL = 1; // 1 or 2 according as ATOL (below) is a scalar or an array
    int* IWORK = new int[LIW];
    double* RWORK = new double[LRW];
    double* y1 = new double[NEQ];

    IOPT = 1;
    for (int i = 0; i < LRW; i++) RWORK[i] = 0;
    for (int i = 0; i < LIW; i++) IWORK[i] = 0;

    double T, TEND;
    double RTOL = 0; // Relative tolerance parameter (scalar)
    double ATOL = 1.0e-8; // Absolute tolerance parameter (scalar or array)

    T = 0; //initial time
    TEND = *dt; //time step
    
    //MOL. CONCENTR.
    for (int i = 0; i < num_gas_species; i++)
    {
        y1[i] = Y1[i] * *Density / phyc.mol_weight[i];
        if (fabs(y1[i]) < ATOL && y1[i] < 0)
        {
            y1[i] = 0;
        }
    }

    y1[num_gas_species] = *Pressure;
    y1[num_gas_species + 1] = *Density;
    y1[num_gas_species + 2] = *Velocity;
    y1[num_gas_species + 3] = *Coordinate;

    ISTATE = 1;
    IWORK[5] = 20000;

    
    CALL_FORTRAN(dlsode,
                init_right_part_H2_multiple_reaction_22, &NEQ, y1, &T, &TEND,
                &ITOL, &RTOL, &ATOL, &ITASK,
                &ISTATE, &IOPT, RWORK, &LRW,
                IWORK, &LIW, init_jacobian, &MF);
    
    
    //cout << *Density << " " << y1[num_gas_species + 1] << endl;
        
    // Mass fractions
    for (int i = 0; i < num_gas_species; i++)
        Y2[i] = y1[i] * phyc.mol_weight[i] / *Density;

    *Pressure = y1[num_gas_species];
    *Density = y1[num_gas_species + 1];
    *Velocity = y1[num_gas_species + 2];
    *Coordinate = y1[num_gas_species + 3];
    *M = sqrt(M2);
    *Thermicity = sigma;
    *SoundSpeed = sqrt(af2);
    
    delete IWORK;
    delete RWORK;
    delete y1;
}


void set_initial_parameters(double *Y, double *Velocity, double *Density, double *Pressure, double *Temperature)
{
    //Mass fractions
    Y[0] = 0.028521857;  //H2
    Y[2] = 0.22636219;  //O2
    Y[8] = 0.745116;    //N2
    
    // Initial temperature
    *Temperature = 295;
    // Initial pressure
    *Pressure = 0.101325;
    double R = get_gas_constant(num_gas_species, Y);
    // Initial density
    *Density = *Pressure / R / *Temperature;
    // Find initial velocity
    CJ_velocity(Y, Velocity, Density, Pressure, Temperature);
    
    return;
}

void znd_structure()
{
    num_gas_species = 23;
    num_react = 70;

    // Mass fractions
    double *Y;
    // Velocity km/s
    double w;
    // Density, kg/m^3
    double rho;
    // Temperature, K
    double T;
    // Pressure, MPa
    double P;
    // Auxiliary Vector
    double *Y2;
    // Coordinate, m
    double x = 0.0;
    // Time, ms
    double t = 0.0;
    // Time step, ms
    double dt = 0.00001;
    // Max number of steps
    int MaxStep = 2000;
    // Mach number
    double M = 0;
    // Thermicity, ms^(-1)
    double sigma = 0;
    // Frozen sound speed, km/s
    double af = 0;

    // In file "out.txt" there will be values of target vector in different moments
    ofstream out("out.txt");
    // File "parameters.txt" will contain parameters at von Neumann point and CJ point
    ofstream parameters("parameters.txt");
    
    // Initialization of physical and chemical constants
    init_consts(num_gas_species, num_react);

    Y = new double[num_gas_species];
    // Initial Mass fractions
    for(int i = 0; i < num_gas_species; i++)
        Y[i] = 0.0;
    
    // Initial Parameters 
    set_initial_parameters(Y, &w, &rho, &P, &T);
    cout << Y[0] << " " << Y[2] << " " << Y[8] << endl;

    // Calculation of initial values of gamma,
    // frozen sound speed and Mach number
    double Cp, Cv;
    Cp = get_Cp(num_gas_species, Y, T);
    Cv = get_Cv(num_gas_species, Y, T);
    // gamma
    gamma = Cp / Cv;
    // frozen sound speed, km/s
    af = sqrt(gamma * P / rho);
    // Mach number
    M = w / af;

    out << w << endl;
/*	out << "Initial " << x << " " << w << " " << rho << " " << P << " " << T << " ";
    for(int j = 0; j < num_species; j++)
        out << Y[j] << " ";
    out << af << " " << M << " " << gamma << " " << sigma << endl;
*/	cout << "Initial values: x = " << x << ", w = " << w << ", rho = " << rho << ", P = " << P << ", T = " << T << ", YH2 = " << Y[0] << endl;
        
    // von Neumann parameters behind the leading shock
    get_vN_parameters(&w, &rho, &P, Y);
    T = P / rho / get_gas_constant(num_gas_species, Y);
    Cp = get_Cp(num_gas_species, Y, T);
    Cv = get_Cv(num_gas_species, Y, T);
    // gamma
    gamma = Cp / Cv;
    // frozen sound speed, km/s
    af = sqrt(gamma * P / rho);
    // Mach number
    M = w / af;

    cout << "von Neumann parameters:" << endl;
    cout << "t = 0" << ", x = 0" << ", w = " << w << ", rho = " << rho << ", P = " << P << ", T = " << T << endl;
    out << t << " " << x << " " << w << " " << rho << " " << P << " " << T << " ";
    for(int j = 0; j < num_gas_species; j++)
        out << Y[j] << " ";
    out << af << " " << M << " " << gamma << " " << sigma << endl;
    
    // Start of numerical integration of differential system
    Y2 = new double[num_gas_species];
    for (int i = 1; (M < 0.99); i++)
    {
        // Integration of Euler equations and chem. kinetics
        integrate(Y, Y2, &P, &rho, &w, &x, &M, &sigma, &af, &dt);
        t += dt;
        
        for(int j = 0; j < num_gas_species; j++)
            Y[j] = Y2[j];
        T = P / rho / get_gas_constant(num_gas_species, Y);

        out << t << " " << x << " " << w << " " << rho << " " << P << " " << T << " ";
            for(int j = 0; j < num_gas_species; j++)
                out << Y[j] << " ";
        out << af << " " << M << " " << gamma << " " << sigma << endl;
        cout << "t = " << i * dt << ", x = " << x << ", w = " << w << ", rho = " << rho << ", P = " << P << ", T = " << T << ", YH2 = " << Y[0] << ", M = " << M << endl;
    }

    delete [] Y;
    delete [] Y2;

    return;
}




#ifndef _CONSTS_H
#define _CONSTS_H

void init_consts(int num_gas_species, int num_react);
double get_Hi(int component_i, double T);
double get_Cpi(int component_i, double T);
double get_Cvi(int component_i, double T);
double get_enthalpy(int num_species, double *Y, double T);
double get_gas_constant(int num_gas_species, double *Y);
double get_Cp(int num_species, double *Y, double T);
double get_Cv(int num_species, double *Y, double T);


struct phy_consts
{
    double kR;  // universal gas constant ( erg/K/mole )
    double kRc; // universal gas constant ( kcal/K/mole )
    double kTime;     // dimensional time ( sec )
    double kLength;   // dimensional length ( cm )
    double kMass;     // dimensional mass ( g )
    double kTemp;      // dimensional temperature ( K )
    double kPres;     // dimensional pressure ( dyn/cm**2 )
    double kDens; // dimensional density ( g/cm**3 )
    double kVel;           // dimensional velocity ( cm/sec )

    // Coefficients for specific heat Cp and
    // molar enthalpy NASA polynoms
    // 8th coef. is for enthalpy calculation
    double** Cp_coef_lT;
    double** Cp_coef_hT;

    // Molar weights
    double* mol_weight;
};

struct che_consts
{
    // Coefficients for Chemical Kinetics
    double* kPrex_f;
    double* kPow_f;
    double* kE_f;
    double* kPrex_f_lp;
    double* kPow_f_lp;
    double* kE_f_lp;
    double* kPrex_r;
    double* kPow_r;
    double* kE_r;
    double* kPrex_r_lp;
    double* kPow_r_lp;
    double* kE_r_lp;
    double* Fcent;
    double* F_f;
    double* F_r;
        
};

extern phy_consts phyc;
extern che_consts chec;

#endif
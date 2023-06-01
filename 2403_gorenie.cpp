// 2403_gorenie.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <fstream>
#include <consts.cpp>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cmath>
using namespace std;

#include <ida/ida.h>   
#include <kinsol/kinsol.h>             /* access to KINSOL func., consts. */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector       */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix       */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype */
#include <sunnonlinsol/sunnonlinsol_newton.h> /* access to Newton SUNNonlinearSolver  */
/* Problem Constants */


#define FTOL   RCONST(1.e-12) /* function tolerance */
#define STOL   RCONST(1.e-12) /* step tolerance     */

#define ZERO   RCONST(0.0)
#define PT25   RCONST(0.25)
#define PT5    RCONST(0.5)
#define ONE    RCONST(1.0)
#define ONEPT5 RCONST(1.5)
#define TWO    RCONST(2.0)

#define PI     RCONST(3.1415926)
#define E      RCONST(2.7182818)

static int num_gas_species = 9;
static int num_react = 22;
int cycle = 0;

double Y_N2 = 0.745187;
double Y_max = 1 - Y_N2;
double P = 0.101325;
double A = 6.85 * pow(10, 12);
double R = 8.314;
double Ea = 0.8 * 46.37 * 293. ;
double koeff_l = 0.35;
double l = 0.5;
long int myiter = 0;
long int nniters;
double eps_x = pow(10, -6);

typedef struct {
   realtype* x;
   realtype* T;
   realtype* Y_H2O;
   int Nx;
   int N_m;
   int NEQ;
   int N_centr;
   realtype Tl;
   realtype M;
   realtype T_center;
   void* mykmem;
} *UserData;


/* Accessor macro */
#define Ith(v,i)    NV_Ith_S(v,i-1)
#define IJth(A,i,j) SM_ELEMENT_D(A,i-1,j-1)

/* Functions Called by the KINSOL Solver */
static int func(N_Vector u, N_Vector f, void* user_data);
static int func_Y(N_Vector u, N_Vector f, void* user_data);
static int resrob(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void* user_data);
static int resrob2(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void* user_data);
static int func_Ydiff(N_Vector u, N_Vector f, void* user_data);

void ExportToArray(vector<double>& T_vect, vector<double>& Y_vect, double& M, UserData data, N_Vector yy, int N_x)
{
    //cout << "data->Tl; = " << data->Tl << "\n";
    T_vect[0] = data->Tl;
    int j = 1;
    for (int i = 1; i < N_x - 1; i++)
    {
        if (i != data->N_centr)
        {
            T_vect[i] = Ith(yy, j - 1 + 1);
            //cout << "T_vect  " << i << " =  " << Ith(yy, j - 1 + 1) << endl;
        }
        else {
            T_vect[i] = data->T_center;
            j--;
        }
        j++;
    }
    T_vect[N_x - 1] = T_vect[N_x - 2];
    //cout << "MyNx = " << myNx << "\n";
    //cout << "MyNm = " << myNm << "\n";
    M = Ith(yy, data->N_m + 1);
    j = 1;
    Y_vect[0] = 0.;
    //cout << "M = " << M << "\n";
    for (int i = data->N_m + 1; i < data->NEQ; i++)
    {
        Y_vect[j] = Ith(yy, i + 1);
        //cout << "Y_vect  " << j << " =  " << Y_vect[j] << endl;
        j++;
    }
    Y_vect[j] = Y_vect[j - 1];
}

void ExportToArray(double* T_vect, double* Y_vect, double& M, UserData data, N_Vector yy, int N_x)
{
    //cout << "data->Tl; = " << data->Tl << "\n";
    T_vect[0] = data->Tl;
    int j = 1;
    for (int i = 1; i < N_x - 1; i++)
    {
        if (i != data->N_centr)
        {
            T_vect[i] = Ith(yy, j - 1 + 1);
            //cout << "T_vect  " << i << " =  " << Ith(yy, j - 1 + 1) << endl;
        }
        else {
            T_vect[i] = data->T_center;
            j--;
        }
        j++;
    }
    T_vect[N_x - 1] = T_vect[N_x - 2];
    //cout << "MyNx = " << myNx << "\n";
    //cout << "MyNm = " << myNm << "\n";
    data->M = Ith(yy, data->N_m + 1);
    j = 1;
    Y_vect[0] = 0.;
    //cout << "M = " << M << "\n";
    for (int i = data->N_m + 1; i < data->NEQ; i++)
    {
        Y_vect[j] = Ith(yy, i + 1);
        //cout << "Y_vect  " << j << " =  " << Y_vect[j] << endl;
        j++;
    }
    Y_vect[j] = Y_vect[j - 1];
}

/* Private Helper Functions */
static int check_retval(void* retvalvalue, const char* funcname, int opt);

static void PrintHeader(realtype rtol, N_Vector avtol, N_Vector y)
{
    realtype* atval, * yval;

    atval = N_VGetArrayPointer(avtol);
    yval = N_VGetArrayPointer(y);

    printf("\nidaRoberts_dns: Robertson kinetics DAE serial example problem for IDA\n");
    printf("         Three equation chemical kinetics problem.\n\n");
    printf("Linear solver: DENSE, with user-supplied Jacobian.\n");
#if defined(SUNDIALS_EXTENDED_PRECISION)
    printf("Tolerance parameters:  rtol = %Lg   atol = %Lg %Lg %Lg \n",
        rtol, atval[0], atval[1], atval[2]);
    printf("Initial conditions y0 = (%Lg %Lg %Lg)\n",
        yval[0], yval[1], yval[2]);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
    printf("Tolerance parameters:  rtol = %g   atol = %g %g %g \n",
        rtol, atval[0], atval[1], atval[2]);
    printf("Initial conditions y0 = (%g %g %g)\n",
        yval[0], yval[1], yval[2]);
#else
    printf("Tolerance parameters:  rtol = %g   atol = %g %g %g \n",
        rtol, atval[0], atval[1], atval[2]);
    printf("Initial conditions y0 = (%g %g %g)\n",
        yval[0], yval[1], yval[2]);
#endif
    printf("Constraints and id not used.\n\n");
    printf("-----------------------------------------------------------------------\n");
    printf("  t             y1           y2           y3");
    printf("      | nst  k      h\n");
    printf("-----------------------------------------------------------------------\n");
}

static void PrintOutput(void* mem, realtype t, N_Vector y)
{
    realtype* yval;
    int retval, kused;
    long int nst;
    realtype hused;

    yval = N_VGetArrayPointer(y);

    retval = IDAGetLastOrder(mem, &kused);
    check_retval(&retval, "IDAGetLastOrder", 1);
    retval = IDAGetNumSteps(mem, &nst);
    check_retval(&retval, "IDAGetNumSteps", 1);
    retval = IDAGetLastStep(mem, &hused);
    check_retval(&retval, "IDAGetLastStep", 1);
#if defined(SUNDIALS_EXTENDED_PRECISION)
    printf("%10.4Le %12.4Le %12.4Le %12.4Le | %3ld  %1d %12.4Le\n",
        t, yval[0], yval[1], yval[2], nst, kused, hused);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
    printf("%10.4e %12.4e %12.4e %12.4e | %3ld  %1d %12.4e\n",
        t, yval[0], yval[1], yval[2], nst, kused, hused);
#else
    printf("%10.4e %12.4e %12.4e %12.4e | %3ld  %1d %12.4e\n",
        t, yval[0], yval[1], yval[2], nst, kused, hused);
#endif
}

static int check_ans(N_Vector y, realtype t, realtype rtol, N_Vector atol)
{
    int      passfail = 0;        /* answer pass (0) or fail (1) retval */
    N_Vector ref;               /* reference solution vector        */
    N_Vector ewt;               /* error weight vector              */
    realtype err;               /* wrms error                       */

    /* create reference solution and error weight vectors */
    ref = N_VClone(y);
    ewt = N_VClone(y);

    /* set the reference solution data */
    NV_Ith_S(ref, 0) = RCONST(5.2083474251394888e-08);
    NV_Ith_S(ref, 1) = RCONST(2.0833390772616859e-13);
    NV_Ith_S(ref, 2) = RCONST(9.9999994791631752e-01);

    /* compute the error weight vector, loosen atol */
    N_VAbs(ref, ewt);
    N_VLinearSum(rtol, ewt, RCONST(10.0), atol, ewt);
    if (N_VMin(ewt) <= ZERO) {
        fprintf(stderr, "\nSUNDIALS_ERROR: check_ans failed - ewt <= 0\n\n");
        return(-1);
    }
    N_VInv(ewt, ewt);

    /* compute the solution error */
    N_VLinearSum(ONE, y, -ONE, ref, ref);
    err = N_VWrmsNorm(ref, ewt);

    /* is the solution within the tolerances? */
    passfail = (err < ONE) ? 0 : 1;

    if (passfail) {
        //fprintf(stdout, "\nSUNDIALS_WARNING: check_ans error=%"GSYM"\n\n", err);
    }

    /* Free vectors */
    N_VDestroy(ref);
    N_VDestroy(ewt);

    return(passfail);
}

void get_Y(double Y_H2O, double& Y_H2, double& Y_O2, double Y_N2)
{
    Y_H2 = (1 - Y_H2O - Y_N2) * 1. / 9.;
    Y_O2 = (1 - Y_H2O - Y_N2) * 8. / 9.;
}

double get_W(double Y_H2O, double Y_H2, double Y_O2, double Y_N2)
{
    double W = Y_H2 / phyc.mol_weight[0] + Y_O2 / phyc.mol_weight[2]
        + Y_H2O / phyc.mol_weight[6] + Y_N2 / phyc.mol_weight[8];
    return 1. / W;
}

double Lambda_H2(double T) {
     double res = (2.0168072101043486 - 1.93742201 * log(T) + 0.29103516 * pow(log(T), 2)
         - 0.00994456 * pow(log(T), 3));
     return  exp(res);
}
double Lambda_O2(double T) {
    double res = (-3.0551023900370224 - 0.85094309 * log(T) + 0.18225493 * pow(log(T), 2)
        - 0.00696862 * pow(log(T), 3));
    return  exp(res);
}
double Lambda_H2O(double T) {
    double res = (11.007354408138713 - 7.62523857 * log(T) + 1.23399258 * pow(log(T), 2)
        - 0.0583371 * pow(log(T), 3));
    return  exp(res);
}
double Lambda_N2(double T) {
    double res = (-3.0551023900370224 - 0.85094309 * log(T) + 0.18225493 * pow(log(T), 2)
        - 0.00696862 * pow(log(T), 3));
    return  exp(res);
}

void Get_mole_fr(double& X_H2O, double& X_H2, double& X_O2, double& X_N2, double Y_H2O, double T)
{
    double Y_H2, Y_O2;
    get_Y(Y_H2O, Y_H2, Y_O2, Y_N2);
    double W = 0;

    W = get_W(Y_H2O, Y_H2, Y_O2, Y_N2);
    X_H2O = Y_H2O * W / phyc.mol_weight[6];
    X_H2 = Y_H2 * W / phyc.mol_weight[0];
    X_O2 = Y_O2 * W / phyc.mol_weight[2];
    X_N2 = Y_N2 * W / phyc.mol_weight[8];
}

double Lambda(double T, double Y_H2O)
{
    double X_H2O, X_H2, X_O2, X_N2;
    Get_mole_fr(X_H2O, X_H2, X_O2, X_N2, Y_H2O, T);

    double lamb1 = X_H2 * Lambda_H2(T) + X_O2 * Lambda_O2(T) 
        + X_H2O * Lambda_H2O(T) + X_N2 * Lambda_N2(T);

    double lamb2 = X_H2 / Lambda_H2(T) + X_O2 / Lambda_O2(T) 
        + X_H2O / Lambda_H2O(T) + X_N2 / Lambda_N2(T);

    return 0.5 * (lamb1 + 1. / lamb2) * pow(10, -2);
}

double Cp_all(double T, double Y_H2O)
{
    double Y_H2, Y_O2;
    get_Y(Y_H2O, Y_H2, Y_O2, Y_N2);
    double cp_tmp = get_Cpi(0, T) * Y_H2 + get_Cpi(2, T) * Y_O2 
        + get_Cpi(6, T) * Y_H2O + get_Cpi(8, T) * Y_N2;
    return cp_tmp;
}

double Cv_all(double T, double Y_H2O)
{
    double Y_H2, Y_O2;
    get_Y(Y_H2O, Y_H2, Y_O2, Y_N2);
    double cp_tmp = get_Cvi(0, T) * Y_H2 + get_Cvi(2, T) * Y_O2
        + get_Cvi(6, T) * Y_H2O + get_Cvi(8, T) * Y_N2;
    return cp_tmp;
}

double Vk_H2O(double Y_H2O, double Y_H2Or, double T, double Tr, realtype* x_cells, const int i)
{
    double h = x_cells[i + 1] - x_cells[i];
    double D = 2.9 * pow(10, -5) * pow((T + Tr) / 2., 0.7);
    double X_H2O, X_H2, X_O2, X_N2;
    Get_mole_fr(X_H2O, X_H2, X_O2, X_N2, Y_H2O, T);
    /*cout << "X_H2O = " << X_H2O << "\n";
    cout << "X_H2 = " << X_H2 << "\n";
    cout << "X_O2 = " << X_O2 << "\n";
    cout << "X_N2 = " << X_N2 << "\n";*/
    double X_H2Or, X_H2r, X_O2r, X_N2r;
    Get_mole_fr(X_H2Or, X_H2r, X_O2r, X_N2r, Y_H2Or, Tr);

    double Y_H2, Y_O2;
    get_Y((Y_H2O + Y_H2Or) / 2., Y_H2, Y_O2, Y_N2);
    double W = get_W((Y_H2O + Y_H2Or) / 2., Y_H2, Y_O2, Y_N2);
    double rho = P * W / phyc.kR / ((Tr + T) / 2.) * pow(10, -3);
    if ((X_H2Or + X_H2O) < eps_x) return 0;
    else return -D / ((X_H2Or + X_H2O) / 2.)  * (X_H2Or - X_H2O) / h;
}

double Vk_H2(double Y_H2O, double Y_H2Or, double T, double Tr, realtype* x_cells, const int i)
{
    double h = x_cells[i + 1] - x_cells[i];
    double D = 2.9 * pow(10, -5) * pow((T + Tr) / 2., 0.7);
    double X_H2O, X_H2, X_O2, X_N2;
    Get_mole_fr(X_H2O, X_H2, X_O2, X_N2, Y_H2O, T);
    double X_H2Or, X_H2r, X_O2r, X_N2r;
    Get_mole_fr(X_H2Or, X_H2r, X_O2r, X_N2r, Y_H2Or, Tr);

    double Y_H2, Y_O2;
    get_Y((Y_H2O + Y_H2Or) / 2., Y_H2, Y_O2, Y_N2);
    double W = get_W((Y_H2O + Y_H2Or) / 2., Y_H2, Y_O2, Y_N2);
    double rho = P * W / phyc.kR / ((Tr + T) / 2.) * pow(10, -3);
    if ((X_H2r + X_H2) < eps_x) return 0;
    else return -D / ((X_H2r + X_H2) / 2.) * (X_H2r - X_H2) / h;
}

double Vk_O2(double Y_H2O, double Y_H2Or, double T, double Tr, realtype* x_cells, const int i)
{
    double h = x_cells[i + 1] - x_cells[i];
    double D = 2.9 * pow(10, -5) * pow((T + Tr) / 2., 0.7);
    double X_H2O, X_H2, X_O2, X_N2;
    Get_mole_fr(X_H2O, X_H2, X_O2, X_N2, Y_H2O, T);
    double X_H2Or, X_H2r, X_O2r, X_N2r;
    Get_mole_fr(X_H2Or, X_H2r, X_O2r, X_N2r, Y_H2Or, Tr);

    double Y_H2, Y_O2;
    get_Y((Y_H2O + Y_H2Or) / 2., Y_H2, Y_O2, Y_N2);
    double W = get_W((Y_H2O + Y_H2Or) / 2., Y_H2, Y_O2, Y_N2);
    double rho = P * W / phyc.kR / ((Tr + T) / 2.) * pow(10, -3);
    if ((X_O2r + X_O2) < eps_x) return 0;
    else return -D / ((X_O2r + X_O2) / 2.) * (X_O2r - X_O2) / h;
}

double Vk_N2(double Y_H2O, double Y_H2Or, double T, double Tr, realtype* x_cells, const int i)
{
    double h = x_cells[i + 1] - x_cells[i];
    double D = 2.9 * pow(10, -5) * pow((T + Tr) / 2., 0.7);
    double X_H2O, X_H2, X_O2, X_N2;
    Get_mole_fr(X_H2O, X_H2, X_O2, X_N2, Y_H2O, T);
    double X_H2Or, X_H2r, X_O2r, X_N2r;
    Get_mole_fr(X_H2Or, X_H2r, X_O2r, X_N2r, Y_H2Or, Tr);

    double Y_H2, Y_O2;
    get_Y((Y_H2O + Y_H2Or) / 2., Y_H2, Y_O2, Y_N2);
    double W = get_W((Y_H2O + Y_H2Or) / 2., Y_H2, Y_O2, Y_N2);
    double rho = P * W / phyc.kR / ((Tr + T) / 2.) * pow(10, -3);
    if ((X_N2r + X_N2) < eps_x) return 0;
    else return -D / ((X_N2r + X_N2) / 2.) * (X_N2r - X_N2) / h;
}

double rhoYkVk_H2O(double Y_H2O, double Y_H2Or, double T, double Tr, realtype* x_cells, const int i)
{
    double h = x_cells[i + 1] - x_cells[i];
    double D = 2.9 * pow(10, -5) * pow((T + Tr) / 2., 0.7);
    double X_H2O, X_H2, X_O2, X_N2;
    Get_mole_fr(X_H2O, X_H2, X_O2, X_N2, Y_H2O, T);
    double X_H2Or, X_H2r, X_O2r, X_N2r;
    Get_mole_fr(X_H2Or, X_H2r, X_O2r, X_N2r, Y_H2Or, Tr);

    double Y_H2, Y_O2;
    get_Y((Y_H2O + Y_H2Or) / 2., Y_H2, Y_O2, Y_N2);
    double W = get_W((Y_H2O + Y_H2Or) / 2., Y_H2, Y_O2, Y_N2);
    double rho = P * W / phyc.kR / ((Tr + T) / 2.) * pow(10, -3);
    return - rho * (phyc.mol_weight[6] * D / W * (X_H2Or - X_H2O) / h);
}

int InitialData(int &Nx, vector<double>& x_vect, vector<double>& T_vect, vector<double>& Y_vect, double& M)
{
    double h = l / (Nx - 1);
    double x_start = koeff_l * l - l / 6.;
    double x_finish = koeff_l * l + l / 6.;
    int dN = (x_finish - x_start) / h;
    double T_start = 293.;
    double T_finish = 2200.;
    double j = 0;
    M = 1000 * 0.000871523;
    cout << "M = " << M << "\n";
    double W;

    for (int i = 0; i < Nx; i++) {
        x_vect[i] = h * i;
    }

    for (int i = 0; i < Nx; i++) {
        T_vect[i] = 1111.31 * tanh((x_vect[i] - koeff_l * l) / 0.038) + 1111.31 + 293.;
    }

    /*for (int i = 0; i < Nx; i++) {
        if (x_vect[i] <= x_start)
        {
            T_vect[i] = T_start;
        }
        else if (x_vect[i] >= x_finish)
        {
            T_vect[i] = T_finish;
        }
        else {
            T_vect[i] = T_vect[i - 1] + (T_finish - T_start) / (dN + 1);
        }
    }*/

    T_vect[0] = 293.;
    j = 0;
    for (int i = 0; i < Nx; i++) {
        if (x_vect[i] <= x_start)
        {
            Y_vect[i] = 0;
        }
        else if (x_vect[i] >= x_finish)
        {
            Y_vect[i] = (1. - Y_N2);
        }
        else {
            Y_vect[i] = (1. - Y_N2) / dN * j;
            j++;
        }
    }
    for (int i = 0; i < Nx - 1; i++) {
        if (x_vect[i] <= koeff_l * l && x_vect[i + 1] > koeff_l * l)
            return i;
    }
}

void Add_elem(vector<double>& T, vector<double>& Y, vector<double>& x, int& N_x, int& N_center, double b)
{
    int j_t = 1;
    double T_max = 0, T_min = T[0];

    for (int i = 0; i < N_x; i++)
    {
        if (T[i] > T_max) T_max = T[i];
        if (T[i] < T_min) T_min = T[i];
    }

    while (j_t < N_x - 2)
    {
        if (fabs(T[j_t] - T[j_t - 1]) > b * (T_max - T_min))
        {
            T.insert(T.begin() + j_t, (T[j_t] + T[j_t - 1]) / 2.);
            Y.insert(Y.begin() + j_t, (Y[j_t] + Y[j_t - 1]) / 2.);
            x.insert(x.begin() + j_t, (x[j_t] + x[j_t - 1]) / 2.);
            N_x++;
            j_t++;
        }
        j_t++;
        //cout << "j_t = " << j_t << "\n";
    }
    T.insert(T.begin(), T[0]);
    Y.insert(Y.begin(), Y[0]);
    x.insert(x.begin(), - x[1]);
    N_x++;

    for (int k = 0; k < 3; k++)
    {
        T.insert(T.begin(), T[1]);
        Y.insert(Y.begin(), Y[1]);
        x.insert(x.begin(), 1.6 * x[0]);
        N_x++;
    }

    for (int i = 0; i < N_x - 1; i++) {
        if (x[i] <= koeff_l * l && x[i + 1] > koeff_l * l)
            N_center = i;
    }
    cout << "N_center = " << N_center << "\n";
    cout << "T N_center = " << T[N_center] << "\n";
}

void Add_elem2(vector<double>& T, vector<double>& Y, vector<double>& x, int& N_x, int& N_center, double b)
{
    int j_t = 1;
    double T_max = 0, T_min = T[0];

    for (int i = 0; i < N_x; i++)
    {
        if (T[i] > T_max) T_max = T[i];
        if (T[i] < T_min) T_min = T[i];
    }

    while (j_t < N_x - 2)
    {
        if (fabs(T[j_t] - T[j_t - 1]) > b * (T_max - T_min))
        {
            T.insert(T.begin() + j_t, (T[j_t] + T[j_t - 1]) / 2.);
            Y.insert(Y.begin() + j_t, (Y[j_t] + Y[j_t - 1]) / 2.);
            x.insert(x.begin() + j_t, (x[j_t] + x[j_t - 1]) / 2.);
            N_x++;
            j_t--;
        }
        j_t++;
        //cout << "j_t = " << j_t << "\n";
    }
    for (int i = 0; i < N_x - 1; i++) {
        if (x[i] <= koeff_l * l && x[i + 1] > koeff_l * l)
            N_center = i;
    }
    cout << "N_center = " << N_center << "\n";
    cout << "T N_center = " << T[N_center] << "\n";
}

double F_right(double T_left, double T_center, double T_right, double M, double Y_H2O, double Y_H2Or, realtype* x_cells, const int i)
{
    double h_left = x_cells[i] - x_cells[i - 1];
    double h = x_cells[i + 1] - x_cells[i];
    double Cp = Cp_all(T_center, Y_H2O) * pow(10, 3);

    //cout << "Cp = " << Cp << "\n";
    //cout << "lambda = " << Lambda((T_right + T_center) / 2., Y_H2O) << "\n";
    double Y_H2, Y_O2;
    get_Y(Y_H2O, Y_H2, Y_O2, Y_N2);
    double W = get_W(Y_H2O, Y_H2, Y_O2, Y_N2);
    //cout << "W = " << W << "\n";
    double rho = P * W / phyc.kR / T_center * pow(10, -3);
    //cout << "rho = " << rho << "\n";
    double K = 1. - Y_N2;
    double Y = 1. - Y_H2O / K;
    double dTdx = (h_left / h / (h + h_left) * T_right + (h - h_left) / h / h_left * T_center - h / h_left / (h + h_left) * T_left);
    //cout << "Hi = " << get_Hi(6, T_center) * pow(10, 3) << endl;
    double w_dot = K * A * rho * rho * Y * exp(-Ea / T_center);
    /*cout << "Lambda = " << -(2. / (h + h_left)) *
        (Lambda((T_right + T_center) / 2., Y_H2O) * (T_right - T_center) / h
            - Lambda((T_center + T_left) / 2., Y_H2O) * (T_center - T_left) / h_left) << "\n";
    cout << "Cp = " << Cp * M * dTdx << "\n";
    cout << "wdot = " << w_dot * get_Hi(6, T_center) * pow(10, 3)
        - w_dot * get_Hi(0, T_center) * pow(10, 3)
        - 0.5 * w_dot * get_Hi(2, T_center) * pow(10, 3) << "\n";
    cout << "diff1 = " << Vk_H2O(Y_H2O, Y_H2Or, T_center, T_right, x_cells, i)<< "\n";
    cout << "diff2 = " << rho * Y_H2 * Vk_H2(Y_H2O, Y_H2Or, T_center, T_right, x_cells, i) * get_Cpi(0, T_center) * dTdx * pow(10, 3) << "\n";
    cout << "diff3 = " << rho * Y_O2 * Vk_O2(Y_H2O, Y_H2Or, T_center, T_right, x_cells, i) * get_Cpi(2, T_center) * dTdx * pow(10, 3) << "\n";
    cout << "diff4 = " <<   rho * Y_N2 * Vk_N2(Y_H2O, Y_H2Or, T_center, T_right, x_cells, i) * get_Cpi(8, T_center) * dTdx * pow(10, 3) << "\n";
    cout << "dT/dx = " << dTdx << "\n";
    cout << "M = " << M << "\n";*/
    //cout << "h = " << (get_Hi(6, 298) - get_Hi(0, 298) - 0.5 * get_Hi(2, 298)) * pow(10, 3) << "\n";
    return -(2. / (h + h_left)) *
        (Lambda((T_right + T_center) / 2., Y_H2O) * (T_right - T_center) / h
            - Lambda((T_center + T_left) / 2., Y_H2O) * (T_center - T_left) / h_left)
        + Cp * M * dTdx
        + w_dot * get_Hi(6, T_center) * pow(10, 3)
        - w_dot * get_Hi(0, T_center) * pow(10, 3)
        - 0.5 * w_dot * get_Hi(2, T_center) * pow(10, 3);
    /*return -(2. / (h + h_left)) *
        (Lambda((T_right + T_center) / 2., Y_H2O) * (T_right - T_center) / h
            - Lambda((T_center + T_left) / 2., Y_H2O) * (T_center - T_left) / h_left)
        + Cp * M * dTdx
        + w_dot * pow(10, 3) * (-5.); */
}

double F_rightdiff(double T_left, double T_center, double T_right, double M, double Y_H2O, double Y_H2Or, realtype* x_cells, const int i)
{
    double h_left = x_cells[i] - x_cells[i - 1];
    double h = x_cells[i + 1] - x_cells[i];
    double Cp = Cp_all(T_center, Y_H2O) * pow(10, 3);
    //cout << "Cp = " << Cp << "\n";
    //cout << "lambda = " << Lambda((T_right + T_center) / 2., Y_H2O) << "\n";
    double Y_H2, Y_O2;
    get_Y(Y_H2O, Y_H2, Y_O2, Y_N2);
    double W = get_W(Y_H2O, Y_H2, Y_O2, Y_N2);
    //cout << "W = " << W << "\n";
    double rho = P * W / phyc.kR / T_center * pow(10, -3);
    //cout << "rho = " << rho << "\n";
    double K = 1. - Y_N2;
    double Y = 1. - Y_H2O / K;
    double dTdx = (h_left / h / (h + h_left) * T_right + (h - h_left) / h / h_left * T_center - h / h_left / (h + h_left) * T_left);
    //cout << "Hi = " << get_Hi(6, T_center) * pow(10, 3) << endl;
    double w_dot = K * A * rho * rho * Y * exp(-Ea / T_center);
    /*cout << "Lambda = " << -(2. / (h + h_left)) *
        (Lambda((T_right + T_center) / 2., Y_H2O) * (T_right - T_center) / h
            - Lambda((T_center + T_left) / 2., Y_H2O) * (T_center - T_left) / h_left) << "\n";
    cout << "Cp = " << Cp * M * dTdx << "\n";
    cout << "wdot = " << w_dot * get_Hi(6, T_center) * pow(10, 3)
        - w_dot * get_Hi(0, T_center) * pow(10, 3)
        - 0.5 * w_dot * get_Hi(2, T_center) * pow(10, 3) << "\n";
    cout << "diff1 = " << Vk_H2O(Y_H2O, Y_H2Or, T_center, T_right, x_cells, i)<< "\n";
    cout << "diff2 = " << rho * Y_H2 * Vk_H2(Y_H2O, Y_H2Or, T_center, T_right, x_cells, i) * get_Cpi(0, T_center) * dTdx * pow(10, 3) << "\n";
    cout << "diff3 = " << rho * Y_O2 * Vk_O2(Y_H2O, Y_H2Or, T_center, T_right, x_cells, i) * get_Cpi(2, T_center) * dTdx * pow(10, 3) << "\n";
    cout << "diff4 = " <<   rho * Y_N2 * Vk_N2(Y_H2O, Y_H2Or, T_center, T_right, x_cells, i) * get_Cpi(8, T_center) * dTdx * pow(10, 3) << "\n";
    cout << "dT/dx = " << dTdx << "\n";
    cout << "M = " << M << "\n";*/
    return -(2. / (h + h_left)) *
        (Lambda((T_right + T_center) / 2., Y_H2O) * (T_right - T_center) / h
            - Lambda((T_center + T_left) / 2., Y_H2O) * (T_center - T_left) / h_left)
        + Cp * M * dTdx
        + w_dot * get_Hi(6, T_center) * pow(10, 3)
        - w_dot * get_Hi(0, T_center) * pow(10, 3)
        - 0.5 * w_dot * get_Hi(2, T_center) * pow(10, 3)
        + rho * Y_H2O * Vk_H2O(Y_H2O, Y_H2Or, T_center, T_right, x_cells, i) * get_Cpi(6, T_center) * dTdx * pow(10, 3)
        + rho * Y_H2 * Vk_H2(Y_H2O, Y_H2Or, T_center, T_right, x_cells, i) * get_Cpi(0, T_center) * dTdx * pow(10, 3)
        + rho * Y_O2 * Vk_O2(Y_H2O, Y_H2Or, T_center, T_right, x_cells, i) * get_Cpi(2, T_center) * dTdx * pow(10, 3)
        + rho * Y_N2 * Vk_N2(Y_H2O, Y_H2Or, T_center, T_right, x_cells, i) * get_Cpi(8, T_center) * dTdx * pow(10, 3);
}

double F_rightY(double T_left, double T_center, double T_right, double M, double Y_H2O_left, double Y_H2O_center, double Y_H2O_right, realtype* x_cells, const int i)
{
    double h_left = x_cells[i] - x_cells[i - 1];
    double h = x_cells[i + 1] - x_cells[i];
    double Y_H2, Y_O2;
    get_Y(Y_H2O_center, Y_H2, Y_O2, Y_N2);
    double W = get_W(Y_H2O_center, Y_H2, Y_O2, Y_N2);
    double rho = P * W / phyc.kR / T_center * pow(10, -3);
    //cout << "rho = " << rho << "\n";
    double K = 1 - Y_N2;
    double Y = 1 - Y_H2O_center / K;
    double w_dot = K * A * rho * rho * Y * exp(-Ea / T_center);
   /* cout << "M = " << M * (h_left / h / (h + h_left) * Y_H2O_right + (h - h_left) / h / h_left * Y_H2O_center
       - h / h_left / (h + h_left) * Y_H2O_left) << "\n";
    cout << "wdot = " << w_dot << "\n";
    cout << "diffusion = " << (rhoYkVk_H2O(Y_H2O_center, Y_H2O_right, T_center, T_right, x_cells, i) - rhoYkVk_H2O(Y_H2O_left, Y_H2O_center, T_left, T_center, x_cells, i - 1))
        / ((x_cells[i + 1] - x_cells[i - 1]) / 2.) << "\n";*/
    return M * (h_left / h / (h + h_left) * Y_H2O_right + (h - h_left) / h / h_left * Y_H2O_center - h / h_left / (h + h_left) * Y_H2O_left) - w_dot;
}

double F_rightYdiff(double T_left, double T_center, double T_right, double M, double Y_H2O_left, double Y_H2O_center, double Y_H2O_right, realtype* x_cells, const int i)
{
    double h_left = x_cells[i] - x_cells[i - 1];
    double h = x_cells[i + 1] - x_cells[i];
    double Y_H2, Y_O2;
    get_Y(Y_H2O_center, Y_H2, Y_O2, Y_N2);
    double W = get_W(Y_H2O_center, Y_H2, Y_O2, Y_N2);
    double rho = P * W / phyc.kR / T_center * pow(10, -3);
    //cout << "rho = " << rho << "\n";
    double K = 1 - Y_N2;
    double Y = 1 - Y_H2O_center / K;
    double w_dot = K * A * rho * rho * Y * exp(-Ea / T_center);
   /* if (myiter < nniters)
    {
        cout << "M = " << M * (h_left / h / (h + h_left) * Y_H2O_right + (h - h_left) / h / h_left * Y_H2O_center
            - h / h_left / (h + h_left) * Y_H2O_left) << "\n";
        cout << "wdot = " << w_dot << "\n";
        cout << "diffusion = " << (rhoYkVk_H2O(Y_H2O_center, Y_H2O_right, T_center, T_right, x_cells, i) - rhoYkVk_H2O(Y_H2O_left, Y_H2O_center, T_left, T_center, x_cells, i - 1))
            / ((x_cells[i + 1] - x_cells[i - 1]) / 2.) << "\n";
    }*/
    return M * (h_left / h / (h + h_left) * Y_H2O_right + (h - h_left) / h / h_left * Y_H2O_center - h / h_left / (h + h_left) * Y_H2O_left) - w_dot
    + (rhoYkVk_H2O(Y_H2O_center, Y_H2O_right, T_center, T_right, x_cells, i) - rhoYkVk_H2O(Y_H2O_left, Y_H2O_center, T_left, T_center, x_cells, i - 1))
     /((x_cells[i + 1] - x_cells[i - 1]) / 2.);
}

int Integrate(int N_x, vector<double>& x_vect, vector<double>& T_vect, vector<double>& Y_vect, double& M, int N_center)
{
    SUNContext sunctx;
    UserData data;
    realtype fnormtol, scsteptol;
    N_Vector res_vect, s, s2, c;
    int glstr, mset, retval;
    void* kmem;
    SUNMatrix J;
    SUNLinearSolver LS;
    int NEQ_T = N_x - 2;
    int NEQ_Y = N_x - 2;
    int NEQ = NEQ_T + NEQ_Y;

    res_vect = NULL;
    s = c = s2 = NULL;
    kmem = NULL;
    J = NULL;
    LS = NULL;
    data = NULL;

    /* Create the SUNDIALS context that all SUNDIALS objects require */
    retval = SUNContext_Create(NULL, &sunctx);
    if (check_retval(&retval, "SUNContext_Create", 1)) return(1);

    /* User data */

    data = (UserData)malloc(sizeof * data);

    /* Create serial vectors of length NEQ */

    res_vect = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)res_vect, "N_VNew_Serial", 0)) return(1);

    s = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)s, "N_VNew_Serial", 0)) return(1);

    s2 = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)s2, "N_VNew_Serial", 0)) return(1);

    c = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)c, "N_VNew_Serial", 0)) return(1);

   //SetInitialGuess1(res_vect, data, NEQ);

    N_VConst(ONE, s); /* no scaling */
    N_VConst(ONE, s2);

    data->Nx = N_x;
    data->x = new realtype[N_x];

    data->Y_H2O = new realtype[N_x];
    data->T = new realtype[N_x];
    data->NEQ = NEQ;
    data->Tl = T_vect[0];
    data->T_center = T_vect[N_center];
    cout << "T_center = " << data->T_center << "\n";
    data->N_centr = N_center;
    int j = 0;
    for (int i = 0; i < N_x; i++) {
        data->x[i] = x_vect[i];
        data->T[i] = T_vect[i];
        data->Y_H2O[i] = Y_vect[i];
        //cout << i << " = " << Ith(res_vect, i + 1) << endl;
    }
    j = 1;
    //cout << NEQ << " = " << Ith(res_vect, NEQ + 1) << endl;
    for (int i = 1; i < NEQ_T; i++) {
        Ith(c, i) = ONE;   /* no constraint on x1 */
        if (j == N_center)
        {
            j++;
        }
        Ith(res_vect, i) = T_vect[j];
        j++;
    }
    Ith(c, NEQ_T) = ONE;
    Ith(res_vect, NEQ_T) = M;
    data->N_m = NEQ_T - 1;
    for (int i = NEQ_T + 1; i <= NEQ; i++) {
        Ith(c, i) = 1.0;   /* constraint on x1 */
        Ith(s, i) = 2500. / 0.25;
        Ith(res_vect, i) = Y_vect[i - NEQ_T];
        //cout << "Yvect " << i - NEQ_T << " =  " << Y_vect[i - NEQ_T] - Y_max << endl;
    }
    fnormtol = FTOL; scsteptol = STOL;


    kmem = KINCreate(sunctx);
    if (check_retval((void*)kmem, "KINCreate", 0)) return(1);

    retval = KINSetUserData(kmem, data);
    if (check_retval(&retval, "KINSetUserData", 1)) return(1);
    retval = KINSetConstraints(kmem, c);
    if (check_retval(&retval, "KINSetConstraints", 1)) return(1);
    retval = KINSetFuncNormTol(kmem, fnormtol);
    if (check_retval(&retval, "KINSetFuncNormTol", 1)) return(1);
    retval = KINSetScaledStepTol(kmem, scsteptol);
    if (check_retval(&retval, "KINSetScaledStepTol", 1)) return(1);
    retval = KINSetMaxSetupCalls(kmem, 1);

    retval = KINInit(kmem, func, res_vect);
    if (check_retval(&retval, "KINInit", 1)) return(1);


    /* Create dense SUNMatrix */
    J = SUNDenseMatrix(NEQ, NEQ, sunctx);
    if (check_retval((void*)J, "SUNDenseMatrix", 0)) return(1);

    /* Create dense SUNLinearSolver object */
    LS = SUNLinSol_Dense(res_vect, J, sunctx);
    if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) return(1);

    /* Attach the matrix and linear solver to KINSOL */
    retval = KINSetLinearSolver(kmem, LS, J);
    if (check_retval(&retval, "KINSetLinearSolver", 1)) return(1);

    glstr = 0;
    mset = 500;

    data->mykmem = kmem;
    retval = KINSol(kmem, res_vect, glstr, s, s2);
    if (check_retval(&retval, "KINSol", 1)) return(1);
    cout << "retval2 = " << retval << "\n";

    int myNx = data->Nx;
    int myNeq = data->NEQ;
    int myNm = data->N_m;
    T_vect[0] = data->Tl;
    ExportToArray(T_vect, Y_vect, M, data, res_vect, N_x);

    /*for (int i = 0; i < N_x; i++) {
        x_vect[i] = data->x[i];
    }
    for (int i = 0; i < N_x; i++) {
        T_vect[i] = data->T[i];
    }
    for (int i = 0; i < N_x; i++) {
        Y_vect[i] = data->Y_H2O[i];
    }
    M = data->M;*/
    /* Free memory */
    printf("\nFinal statsistics:\n");
    retval = KINPrintAllStats(kmem, stdout, SUN_OUTPUTFORMAT_TABLE);
    N_VDestroy(res_vect);
    N_VDestroy(s);
    N_VDestroy(c);
    KINFree(&kmem);
    SUNLinSolFree(LS);
    SUNMatDestroy(J);
    free(data);
    SUNContext_Free(&sunctx);
    return 0;
}

int Integrate_Y(int N_x, vector<double>& x_vect, vector<double>& T_vect, vector<double>& Y_vect, double& M, int N_center, int diff)
{
    SUNContext sunctx;
    UserData data;
    realtype fnormtol, scsteptol;
    N_Vector res_vect, s, c;
    int glstr, mset, retval;
    void* kmem;
    SUNMatrix J;
    SUNLinearSolver LS;
    //int NEQ_T = N_x - 2;
    int NEQ_Y = N_x - 2;
    int NEQ = 1 + NEQ_Y;

    res_vect = NULL;
    s = c = NULL;
    kmem = NULL;
    J = NULL;
    LS = NULL;
    data = NULL;

    /* Create the SUNDIALS context that all SUNDIALS objects require */
    retval = SUNContext_Create(NULL, &sunctx);
    if (check_retval(&retval, "SUNContext_Create", 1)) return(1);

    /* User data */

    data = (UserData)malloc(sizeof * data);

    /* Create serial vectors of length NEQ */

    res_vect = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)res_vect, "N_VNew_Serial", 0)) return(1);

    s = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)s, "N_VNew_Serial", 0)) return(1);

    c = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)c, "N_VNew_Serial", 0)) return(1);

    N_VConst(ONE, s); /* no scaling */

    data->Nx = N_x;
    data->x = new realtype[N_x];

    data->Y_H2O = new realtype[N_x];
    data->T = new realtype[N_x];
    data->NEQ = NEQ;
    data->Tl = T_vect[0];
    data->T_center = T_vect[N_center];
    cout << "T_center = " << data->T_center << "\n";
    data->N_centr = N_center;
    int j = 0;
    for (int i = 0; i < N_x; i++) {
        data->x[i] = x_vect[i];
        data->T[i] = T_vect[i];
        data->Y_H2O[i] = Y_vect[i];
        //cout << i << " = " << Ith(res_vect, i + 1) << endl;
    }

    Ith(c, 1) = ONE;
    Ith(res_vect, 1) = M;
    data->N_m = 1 - 1;

    for (int i = 2; i <= NEQ; i++) {
        Ith(c, i) = ONE;   /* no constraint on x1 */
        Ith(res_vect, i) = Y_vect[i - 1];
        //cout << "Yvect " << i - N_x + 1 << " =  " << Y_vect[i - N_x + 1] << endl;
    }
    fnormtol = FTOL; scsteptol = STOL;


    kmem = KINCreate(sunctx);
    if (check_retval((void*)kmem, "KINCreate", 0)) return(1);

    retval = KINSetUserData(kmem, data);
    if (check_retval(&retval, "KINSetUserData", 1)) return(1);
    retval = KINSetConstraints(kmem, c);
    if (check_retval(&retval, "KINSetConstraints", 1)) return(1);
    retval = KINSetFuncNormTol(kmem, fnormtol);
    if (check_retval(&retval, "KINSetFuncNormTol", 1)) return(1);
    retval = KINSetScaledStepTol(kmem, scsteptol);
    if (check_retval(&retval, "KINSetScaledStepTol", 1)) return(1);
    
    if (diff == 0) {
        cout << "func_Y " << "\n";
        retval = KINInit(kmem, func_Y, res_vect);
        if (check_retval(&retval, "KINInit", 1)) return(1);
    }
    if (diff == 1) {
        cout << "func_Ydiff" << "\n";
        retval = KINInit(kmem, func_Ydiff, res_vect);
        if (check_retval(&retval, "KINInit", 1)) return(1);
    }

    /* Create dense SUNMatrix */
    J = SUNDenseMatrix(NEQ, NEQ, sunctx);
    if (check_retval((void*)J, "SUNDenseMatrix", 0)) return(1);

    /* Create dense SUNLinearSolver object */
    LS = SUNLinSol_Dense(res_vect, J, sunctx);
    if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) return(1);

    /* Attach the matrix and linear solver to KINSOL */
    retval = KINSetLinearSolver(kmem, LS, J);
    if (check_retval(&retval, "KINSetLinearSolver", 1)) return(1);

    glstr = 0;
    mset = 500;

    data->mykmem = kmem;
    retval = KINSol(kmem, res_vect, glstr, s, s);
    if (check_retval(&retval, "KINSol", 1)) return(1);
    //PrintFinalStats(kmem);
    cout << "retval = " << retval << "\n";

    int myNx = data->Nx;
    int myNeq = data->NEQ;
    int myNm = data->N_m;

    T_vect[0] = data->Tl;
    //cout << "MyNx = " << myNx << "\n";
    //cout << "MyNm = " << myNm << "\n";
    M = Ith(res_vect, myNm + 1);
    Y_vect[0] = 0.;
    cout << "M = " << M << "\n";
    for (int i = myNm + 1; i < N_x - 1; i++)
    {
        Y_vect[i] = Ith(res_vect, i + 1);
        //cout << "Y_vect  " << i << " =  " << Y_vect[i] << endl;
    }
    Y_vect[N_x - 1] = Y_vect[N_x - 2];
    /* Free memory */
    printf("\nFinal statsistics:\n");
    retval = KINPrintAllStats(kmem, stdout, SUN_OUTPUTFORMAT_TABLE);
    N_VDestroy(res_vect);
    N_VDestroy(s);
    N_VDestroy(c);
    KINFree(&kmem);
    SUNLinSolFree(LS);
    SUNMatDestroy(J);
    free(data);
    SUNContext_Free(&sunctx);
    return 0;
}

int Integrate_IDA(int N_x, vector<double>& x_vect, vector<double>& T_vect, vector<double>& Y_vect, double& M, int N_center)
{
    void* mem;
    N_Vector yy, yp, avtol;
    realtype rtol, * yval, * ypval, * atval;
    realtype t0, tout1, tout, tret;
    int iout, retval, retvalr;
    SUNMatrix A;
    SUNLinearSolver LS;
    SUNNonlinearSolver NLS;
    SUNContext ctx;
    UserData data;
    data = (UserData)malloc(sizeof * data);

    int NEQ_T = N_x - 2;
    int NEQ_Y = N_x - 2;
    int NEQ = NEQ_T + NEQ_Y;


    data->Nx = N_x;
    data->x = new realtype[N_x];
    data->Y_H2O = new realtype[N_x];
    data->T = new realtype[N_x];
    data->NEQ = NEQ;
    data->Tl = T_vect[0];
    data->T_center = T_vect[N_center];
    data->N_centr = N_center;
    data->M = M;

    int j = 0;
    for (int i = 0; i < N_x; i++) {
        data->x[i] = x_vect[i];
        data->T[i] = T_vect[i];
        data->Y_H2O[i] = Y_vect[i];
        //cout << i << " = " << Ith(res_vect, i + 1) << endl;
    }

    mem = NULL;
    yy = yp = avtol = NULL;
    yval = ypval = atval = NULL;
    A = NULL;
    LS = NULL;
    NLS = NULL;
    /* Create SUNDIALS context */
    retval = SUNContext_Create(NULL, &ctx);
    if (check_retval(&retval, "SUNContext_Create", 1)) return(1);

    /* Allocate N-vectors. */
    yy = N_VNew_Serial(NEQ, ctx);
    if (check_retval((void*)yy, "N_VNew_Serial", 0)) return(1);
    yp = N_VClone(yy);
    if (check_retval((void*)yp, "N_VNew_Serial", 0)) return(1);
    avtol = N_VClone(yy);
    if (check_retval((void*)avtol, "N_VNew_Serial", 0)) return(1);

    /* Create and initialize  y, y', and absolute tolerance vectors. */
    yval = N_VGetArrayPointer(yy);
    ypval = N_VGetArrayPointer(yp);
    rtol = RCONST(1.0e-6);
    atval = N_VGetArrayPointer(avtol);
    /* Integration limits */
    t0 = ZERO;
    tout1 = pow(10, -6);

    j = 1;
    //cout << NEQ << " = " << Ith(res_vect, NEQ + 1) << endl;
    for (int i = 1; i < NEQ_T; i++) {
        Ith(avtol, i) = RCONST(1.0e-8);
        if (j == N_center) j++;
        Ith(yy, i) = T_vect[j];
        //cout << "yy " << i << " = " << Ith(yy, i) << "\n";
        j++;
    }

    Ith(avtol, NEQ_T) = RCONST(1.0e-12);
    Ith(yy, NEQ_T) = M;
    data->N_m = NEQ_T - 1;
    //cout << "yy " << NEQ_T << " = " << Ith(yy, NEQ_T) << "\n";

    for (int i = NEQ_T + 1; i <= NEQ; i++) {
        Ith(avtol, i) = RCONST(1.0e-12);
        Ith(yy, i) = Y_vect[i - NEQ_T];
        //cout << "yy " << i << " = " << Ith(yy, i) << "\n";
    }

    j = 1;
    for (int i = 1; i < NEQ_T; i++) {
        if (j == N_center) j++;
        Ith(yp, i) = -F_right(T_vect[j - 1], T_vect[j], T_vect[j + 1], data->M, Y_vect[j], Y_vect[j + 1], data->x, j);
        //cout << "Ithypi = " << -F_right(T_vect[j - 1], T_vect[j], T_vect[j + 1], data->M, Y_vect[j], Y_vect[j + 1], data->x, j) << "\n";
        j++;

    }
    Ith(yp, NEQ_T) = 0;
    for (int i = NEQ_T + 1; i <= NEQ; i++) {
        j = i - NEQ_T;
        Ith(yp, i) = - F_rightY(T_vect[j - 1], T_vect[j], T_vect[j + 1], data->M, Y_vect[j - 1], Y_vect[j], Y_vect[j + 1], data->x, j);
    }

    /* Call IDACreate and IDAInit to initialize IDA memory */
    mem = IDACreate(ctx);
    if (check_retval((void*)mem, "IDACreate", 0)) return(1);

    retval = IDAInit(mem, resrob, t0, yy, yp);
    if (check_retval(&retval, "IDAInit", 1)) return(1);
    /* Call IDASVtolerances to set tolerances */

    retval = IDASVtolerances(mem, rtol, avtol);
    if (check_retval(&retval, "IDASVtolerances", 1)) return(1);

    retval = IDASetUserData(mem, data);
    if (check_retval(&retval, "IDASetUserData", 1)) return(1);
    retval = IDASetMaxNumSteps(mem, 20000);

    /* Create dense SUNMatrix for use in linear solves */
    A = SUNDenseMatrix(NEQ, NEQ, ctx);
    if (check_retval((void*)A, "SUNDenseMatrix", 0)) return(1);

    /* Create dense SUNLinearSolver object */
    LS = SUNLinSol_Dense(yy, A, ctx);
    if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) return(1);

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolver(mem, LS, A);
    if (check_retval(&retval, "IDASetLinearSolver", 1)) return(1);
   
    NLS = SUNNonlinSol_Newton(yy, ctx);
    if (check_retval((void*)NLS, "SUNNonlinSol_Newton", 0)) return(1);

    /* Attach the nonlinear solver */
    retval = IDASetNonlinearSolver(mem, NLS);
    if (check_retval(&retval, "IDASetNonlinearSolver", 1)) return(1);

    /* In loop, call IDASolve, print results, and test for error.
       Break out of loop when NOUT preset output times have been reached. */

    iout = 0; tout = tout1;
    double tend = pow(10, 5);
    ofstream fout;
    double Y_H2, Y_O2;
    double W, w_dot, rho;
    while (iout  < 10) {
        retval = IDASolve(mem, tout, &tret, yy, yp, IDA_NORMAL);
        ExportToArray(T_vect, Y_vect, M, data, yy, N_x);
        //PrintOutput(mem, tret, yy);
        cout << "t = " << tout << "\n";
        cout << "M = " << M << "\n";
        j = 1;
        for (int i = 1; i < NEQ_T; i++) {
            if (j == N_center) j++;
            //cout << "Ithyp " << i - 1 << " = " << -F_right(T_vect[j - 1], T_vect[j], T_vect[j + 1], data->M, Y_vect[j], Y_vect[j + 1], data->x, j) << "\n";
            j++;

        }
        Ith(yp, NEQ_T) = 0;
        for (int i = NEQ_T + 1; i <= NEQ; i++) {
            j = i - NEQ_T;
            //cout << "Ithyp " << i - 1 << " = " << -F_right(T_vect[j - 1], T_vect[j], T_vect[j + 1], data->M, Y_vect[j], Y_vect[j + 1], data->x, j) << "\n";
        }
        fout.open("file" + to_string(tout * pow(10, 8)) + ".dat");
        fout << "TITLE=\"" << "Graphics" << "\"" << endl;
        fout << R"(VARIABLES= "x", "T", "Y fr", "rho")" << endl;
        for (int i = 0; i < N_x; i++) {
            get_Y(Y_vect[i], Y_H2, Y_O2, Y_N2);
            W = get_W(Y_vect[i], Y_H2, Y_O2, Y_N2);
            rho = P * W / phyc.kR / T_vect[i] * pow(10, -3);
            double K = 1. - Y_N2;
            double Y = 1. - Y_vect[i] / K;
            fout << x_vect[i] << "  " << T_vect[i] << " " << Y_vect[i] << " " << rho << endl;
        }
        fout.close();
        if (check_retval(&retval, "IDASolve", 1)) return(1);

        if (retval == IDA_SUCCESS) {
            iout++;
            tout += tout1;
        }
    }

    /* Print final statistics to the screen */
    printf("\nFinal Statistics:\n");
    retval = IDAPrintAllStats(mem, stdout, SUN_OUTPUTFORMAT_TABLE);

    /* Print final statistics to a file in CSV format */
    //FID = fopen("idaRoberts_dns_stats.csv", "w");
    //retval = IDAPrintAllStats(mem, FID, SUN_OUTPUTFORMAT_CSV);
    //fclose(FID);

    /* check the solution error */
    //retval = check_ans(yy, tret, rtol, avtol);

    /* Free memory */
    IDAFree(&mem);
    SUNNonlinSolFree(NLS);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
    N_VDestroy(avtol);
    N_VDestroy(yy);
    N_VDestroy(yp);
    SUNContext_Free(&ctx);

    return(retval);
}

static int func_T(N_Vector u, N_Vector f, void* user_data)
{
    realtype* T, * fdata;
    T = N_VGetArrayPointer(u);
    fdata = N_VGetArrayPointer(f);
    double Tl = 293;
    double Y_O2, Y_H2, Y_H2O;
    Y_H2O = 1 - Y_N2;
    get_Y(0, Y_H2, Y_O2, Y_N2);
    cout << "Y_H2 = " << Y_H2 << "\n";
    cout << "Y_O2 = " << Y_O2 << "\n";
    cout << "Y_N2 = " << Y_N2 << "\n";
    cout << " 1 = " << Y_H2 + Y_O2 + Y_N2 << "\n";
    cout << "get_Hi(8, Tl) = " << get_Hi(8, Tl) << "\n";
    cout << "get_Hi(0, Tl) = " << get_Hi(0, Tl) << "\n";
    cout << "get_Hi(2, Tl) = " << get_Hi(2, Tl) << "\n";
    cout << "get_Hi(6, Tl) = " << get_Hi(6, Tl) << "\n";
    cout << "get_Hi(6, T[0]) = " << get_Hi(6, 2515) << "\n";
    cout << "get_Hi(8, T[0]) = " << get_Hi(8, 2515) << "\n";
    fdata[0] = Y_N2 * get_Hi(8, T[0]) + Y_H2O * get_Hi(6, T[0]) 
        - Y_H2 * get_Hi(0, Tl) - Y_O2 * get_Hi(2, Tl) - Y_N2 * get_Hi(8, Tl);
    
    //cout << "fdata = " << fdata[0] << endl;
    //cout << "Hi1 = " << get_Hi(6, T[0]) << endl;
    cout << "R = " << phyc.kR << "\n";
    cout << "mol = " << phyc.mol_weight[6] << "\n";
    //cout << "H = " << get_Hi(6, T[0]) - 8. / 9. * get_Hi(2, Tl) - 1. / 9. * get_Hi(0, Tl) << endl;
    //cout << "molweight = " << phyc.mol_weight[6] << endl;
    //cout << "Q = " << Q << endl;
    return(0);
}

double T_find()
{
    SUNContext sunctx;
    UserData data;
    realtype fnormtol, scsteptol;
    N_Vector res_vect, s, c;
    int glstr, mset, retval;
    void* kmem;
    SUNMatrix J;
    SUNLinearSolver LS;
    double NEQ = 1;
    res_vect = NULL;
    s = c = NULL;
    kmem = NULL;
    J = NULL;
    LS = NULL;
    data = NULL;

    /* Create the SUNDIALS context that all SUNDIALS objects require */
    retval = SUNContext_Create(NULL, &sunctx);
    if (check_retval(&retval, "SUNContext_Create", 1)) return(1);

    /* Create serial vectors of length NEQ */

    res_vect = N_VNew_Serial(NEQ, sunctx);

    if (check_retval((void*)res_vect, "N_VNew_Serial", 0)) return(1);

    s = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)s, "N_VNew_Serial", 0)) return(1);

    c = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)c, "N_VNew_Serial", 0)) return(1);

    //SetInitialGuess1(res_vect, data, NEQ);

    N_VConst(ONE, s); /* no scaling */
    Ith(res_vect, 1) = 293;
    Ith(c, 1) = ONE;
    fnormtol = FTOL; scsteptol = STOL;


    kmem = KINCreate(sunctx);
    if (check_retval((void*)kmem, "KINCreate", 0)) return(1);

    retval = KINSetUserData(kmem, data);
    if (check_retval(&retval, "KINSetUserData", 1)) return(1);
    retval = KINSetConstraints(kmem, c);
    if (check_retval(&retval, "KINSetConstraints", 1)) return(1);
    retval = KINSetFuncNormTol(kmem, fnormtol);
    if (check_retval(&retval, "KINSetFuncNormTol", 1)) return(1);
    retval = KINSetScaledStepTol(kmem, scsteptol);
    if (check_retval(&retval, "KINSetScaledStepTol", 1)) return(1);

    retval = KINInit(kmem, func_T, res_vect);
    if (check_retval(&retval, "KINInit", 1)) return(1);

    /* Create dense SUNMatrix */
    J = SUNDenseMatrix(NEQ, NEQ, sunctx);
    if (check_retval((void*)J, "SUNDenseMatrix", 0)) return(1);

    /* Create dense SUNLinearSolver object */
    LS = SUNLinSol_Dense(res_vect, J, sunctx);
    if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) return(1);

    /* Attach the matrix and linear solver to KINSOL */
    retval = KINSetLinearSolver(kmem, LS, J);
    if (check_retval(&retval, "KINSetLinearSolver", 1)) return(1);

    /* Print out the problem size, solution parameters, initial guess. */
    //PrintHeader(fnormtol, scsteptol);

    /* --------------------------- */

    glstr = 0;
    retval = KINSol(kmem, res_vect, glstr, s, s);
    if (check_retval(&retval, "KINSol", 1)) return(1);
    cout << "T res = " << Ith(res_vect, 1) << endl;
    //PrintFinalStats(kmem);
    /* Free memory */
    printf("\nFinal statsistics:\n");
    retval = KINPrintAllStats(kmem, stdout, SUN_OUTPUTFORMAT_TABLE);
    N_VDestroy(res_vect);
    N_VDestroy(s);
    N_VDestroy(c);
    KINFree(&kmem);
    SUNLinSolFree(LS);
    SUNMatDestroy(J);
    free(data);
    SUNContext_Free(&sunctx);
    return 0;
}

int main()
{
    init_consts(num_gas_species, num_react);
    int N_x = 100;
    double b = 0.01;
    double M;
    double W, rho, Y_H2, Y_O2;
    int N_center;
    int retval;
    double w_dot;
    vector<double> x_vect(N_x);
    vector<double> T_vect(N_x);
    vector<double> Y_vect(N_x);
    double* my_x;
    ofstream fout;
    N_center = InitialData(N_x, x_vect, T_vect, Y_vect, M);
    cout << "N_center = " << N_center << "\n";
    //T_find();
    //cout << "endT\n";
    Add_elem(T_vect, Y_vect, x_vect, N_x, N_center, b);
    cout << "N_x = " << N_x << "\n";
    cout << "N_center = " << N_center << "\n";
    my_x = new double[N_x];
    for (int i = 0; i < N_x; i++) {
        my_x[i] = x_vect[i];
    }
    cout << "F = " << F_right(T_vect[N_center - 1], T_vect[N_center], T_vect[N_center + 1], M, Y_vect[N_center], Y_vect[N_center + 1], my_x, N_center) << "\n";


    retval = Integrate_Y(N_x, x_vect, T_vect, Y_vect, M, N_center, 0);
    fout.open("fileY1_" + to_string(N_x) + ".dat");
    fout << "TITLE=\"" << "Graphics" << "\"" << endl;
    fout << R"(VARIABLES= "x", "T1", "Y fr1", "rho1", "wdot1")" << endl;
    for (int i = 0; i < N_x; i++) {
        get_Y(Y_vect[i], Y_H2, Y_O2, Y_N2);
        W = get_W(Y_vect[i], Y_H2, Y_O2, Y_N2);
        rho = P * W / phyc.kR / T_vect[i] * pow(10, -3);
        double K = 1. - Y_N2;
        double Y = 1. - Y_vect[i] / K;
        w_dot = K * A * rho * rho * Y * exp(-Ea / T_vect[i]);
        fout << x_vect[i] << "  " << T_vect[i] << " " << Y_vect[i] << " " << rho << " " << w_dot << endl;
    }
    fout.close();


    //retval = Integrate_Y(N_x, x_vect, T_vect, Y_vect, M, N_center, 1);
    retval = Integrate_IDA(N_x, x_vect, T_vect, Y_vect, M, N_center);
    fout.open("fileY2_" + to_string(N_x) + ".dat");
    fout << "TITLE=\"" << "Graphics" << "\"" << endl;
    fout << R"(VARIABLES= "x", "T2", "Y fr2", "rho2", "Cp", "gamma")" << endl;
    for (int i = 0; i < N_x; i++) {
        get_Y(Y_vect[i], Y_H2, Y_O2, Y_N2);
        W = get_W(Y_vect[i], Y_H2, Y_O2, Y_N2);
        rho = P * W / phyc.kR / T_vect[i] * pow(10, -3);
        double K = 1. - Y_N2;
        double Y = 1. - Y_vect[i] / K;
        w_dot = K * A * rho * rho * Y * exp(-Ea / T_vect[i]);
        fout << x_vect[i] << "  " << T_vect[i] << " " << Y_vect[i] << " " << rho << " " << Cp_all(T_vect[i], Y_vect[i]) * pow(10, 3) << " " << Cp_all(T_vect[i], Y_vect[i])  / Cv_all(T_vect[i], Y_vect[i]) << endl;
    }
    fout.close();
    cout << "MY = " << M << endl;


    //retval = Integrate(N_x, x_vect, T_vect, Y_vect, M, N_center);
    //cout << "MT = " << M << endl;
    //fout.open("fileTY" + to_string(N_x) + ".dat");
    //fout << "TITLE=\"" << "Graphics" << "\"" << endl;
    //fout << R"(VARIABLES= "x", "T2", "Y fr2", "rho2", "wdot")" << endl;
    //for (int i = 0; i < N_x; i++) {
    //    get_Y(Y_vect[i], Y_H2, Y_O2, Y_N2);
    //    W = get_W(Y_vect[i], Y_H2, Y_O2, Y_N2);
    //    rho = P * W / phyc.kR / T_vect[i] * pow(10, -3);
    //    double K = 1. - Y_N2;
    //    double Y = 1. - Y_vect[i] / K;
    //    w_dot = K * A * rho * rho * Y * exp(-Ea / T_vect[i]);
    //    fout << x_vect[i] << "  " << T_vect[i] << " " << Y_vect[i] << " " << rho << " " << w_dot << endl;
    //}
    //fout.close();
    delete[] my_x;
    return 0;
    //T_find();
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY KINSOL
 *--------------------------------------------------------------------
 */

 /*
  * System function for predator-prey system
  */


static int func(N_Vector u, N_Vector f, void* user_data)
{
    realtype* T, * fdata;
    realtype x1, l1, L1, x2, l2, L2;
    realtype* x_cells, * T_vect, * Y_vect;
    UserData data;
    double h_left, h;
    double tmp;
    double M;
    data = (UserData)user_data;
    x_cells = data->x;
    T_vect = data->T;
    Y_vect = data->Y_H2O;
    int myNx = data->Nx;
    int myNeq = data->NEQ;
    int myNm = data->N_m;

    T = N_VGetArrayPointer(u);
    fdata = N_VGetArrayPointer(f);
    double T_cer = data->T_center;
    int j = 0;
    //настоящий вектор T
    ExportToArray(T_vect, Y_vect, M, data, u, myNx);
    //cout << "Y_vect  " << j << " =  " << Y_vect[j] << endl;
    int fl = 0;
    //cout << "cycle = " << cycle << "\n";
    long int nniters;
    int retval = KINGetNumNonlinSolvIters(data->mykmem, &nniters);
    ofstream foutT;
    ofstream foutY;
    /*if (myiter < nniters)
    {
        foutT.open(to_string(nniters) + "T.dat");
        foutT << "TITLE=\"" << "Graphics" << "\"" << endl;
        foutT << R"(VARIABLES= "N", "FT")" << endl;
        cout << "nniters = " << nniters << "\n";
        foutY.open(to_string(nniters) + "Y.dat");
        foutY << "TITLE=\"" << "Graphics" << "\"" << endl;
        foutY << R"(VARIABLES= "N", "FY")" << endl;
        cout << "nniters = " << nniters << "\n";
    }*/

    for (j = 1; j < myNx - 1; j++) {
        fdata[j - 1] = F_right(T_vect[j - 1], T_vect[j], T_vect[j + 1], data->M, Y_vect[j], Y_vect[j + 1],x_cells, j);
        //cout << j - 1 << " " << fdata[j - 1] << "\n";
        //if (myiter < nniters)
        //{
        //    foutT << j - 1 << " " << fdata[j - 1] << "\n";
        //    //cout << j - 1 << " " << fdata[j - 1] << "\n";
        //}
    }
    //cout << "next\n";
    for (int i = 1; i < myNx - 1; i++) {
        fdata[j - 1] = F_rightY(T_vect[i - 1], T_vect[i], T_vect[i + 1], data->M, Y_vect[i - 1], Y_vect[i], Y_vect[i + 1], x_cells, i);
        //cout << "j - 1 = " << j - 1 << "\n";
        /*if (myiter < nniters)
        {
            foutY << i - 1 << " " << fdata[j - 1] << "\n";
        }*/
        j++;
    }
    cycle++;
   /* if (myiter < nniters)
    {
        myiter++;
        foutT.close();
        foutY.close();
    }*/
    //cout << "\n" << "\n" << "\n" << "\n" << "\n" << "\n";
    return(0);
}

static int func_Y(N_Vector u, N_Vector f, void* user_data)
{
    realtype* Y, * fdata;
    realtype x1, l1, L1, x2, l2, L2;
    realtype* x_cells, * T_vect, * Y_vect;
    UserData data;
    double h_left, h;
    double tmp;
    double M;
    data = (UserData)user_data;
    x_cells = data->x;
    T_vect = data->T;
    Y_vect = data->Y_H2O;
    int myNx = data->Nx;
    int myNeq = data->NEQ;
    int myNm = data->N_m;

    Y = N_VGetArrayPointer(u);
    fdata = N_VGetArrayPointer(f);
    double T_cer = data->T_center;

    //cout << "MyNx = " << myNx << "\n";
    //cout << "MyNm = " << myNm << "\n";
    data->M = Y[myNm];
    int j = 1;
    //cout << "M = " << data->M << "\n";
    for (int i = myNm + 1; i < myNx - 1; i++)
    {
        Y_vect[i] = Y[i];
        //cout << "Y_vect5555  " << i << " =  " << Y_vect[i] << endl;
    }
    Y_vect[0] = 0.;
    Y_vect[myNx - 1] = Y_vect[myNx - 2];
    int fl = 0;
    int Ncentr = data->N_centr;

    fdata[0] = F_right(T_vect[Ncentr - 1], T_vect[Ncentr], T_vect[Ncentr + 1], data->M, Y_vect[Ncentr], Y_vect[Ncentr + 1], x_cells, Ncentr);
    //cout << "fdata " << 0 << " = " << fdata[0] << "\n";
    //cout << "next\n";

    for (int i = 1; i < myNeq; i++) {
        fdata[i] = F_rightY(T_vect[i - 1], T_vect[i], T_vect[i + 1], data->M, Y_vect[i - 1], Y_vect[i], Y_vect[i + 1], x_cells, i);
        //cout << "j - 1 = " << j - 1 << "\n";
        //cout << "fdata " << i << " = " << fdata[i] << "\n";
    }
    //cycle++;
    //cout << "cycle = " << cycle << "\n";
    //cout << "\n" << "\n" << "\n" << "\n" << "\n" << "\n";
    return(0);
}

static int func_Ydiff(N_Vector u, N_Vector f, void* user_data)
{
    realtype* Y, * fdata;
    realtype x1, l1, L1, x2, l2, L2;
    realtype* x_cells, * T_vect, * Y_vect;
    UserData data;
    double h_left, h;
    double tmp;
    double M;
    data = (UserData)user_data;
    x_cells = data->x;
    T_vect = data->T;
    Y_vect = data->Y_H2O;
    int myNx = data->Nx;
    int myNeq = data->NEQ;
    int myNm = data->N_m;

    Y = N_VGetArrayPointer(u);
    fdata = N_VGetArrayPointer(f);
    double T_cer = data->T_center;

    //cout << "MyNx = " << myNx << "\n";
    //cout << "MyNm = " << myNm << "\n";
    data->M = Y[myNm];
    int j = 1;
    //cout << "M = " << data->M << "\n";
    for (int i = myNm + 1; i < myNx - 1; i++)
    {
        Y_vect[i] = Y[i];
        //cout << "Y_vect5555  " << i << " =  " << Y_vect[i] << endl;
    }
    Y_vect[0] = 0.;
    Y_vect[myNx - 1] = Y_vect[myNx - 2];
    int fl = 0;
    int Ncentr = data->N_centr;
    int retval = KINGetNumNonlinSolvIters(data->mykmem, &nniters);
    /*if (myiter < nniters)
    {
        cout << "\n\n\n\n\n\n nniters = " << nniters << "\n\n\n\n\n\n\n";
    }*/
    fdata[0] = F_rightdiff(T_vect[Ncentr - 1], T_vect[Ncentr], T_vect[Ncentr + 1], data->M, Y_vect[Ncentr], Y_vect[Ncentr + 1], x_cells, Ncentr);
    //cout << "fdata " << 0 << " = " << fdata[0] << "\n";
    //cout << "next\n";

    for (int i = 1; i < myNeq; i++) {
        fdata[i] = F_rightYdiff(T_vect[i - 1], T_vect[i], T_vect[i + 1], data->M, Y_vect[i - 1], Y_vect[i], Y_vect[i + 1], x_cells, i);
        //cout << "j - 1 = " << j - 1 << "\n";
        if (myiter < nniters)
        {
            //cout << "fdata " << i << " = " << fdata[i] << "\n";
        }
    }
    if (myiter < nniters)
    {
        myiter++;
    }

    //cycle++;
    //cout << "cycle = " << cycle << "\n";
    //cout << "\n" << "\n" << "\n" << "\n" << "\n" << "\n";
    return(0);
}

static int resrob(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void* user_data)
{
    realtype* yval, * ypval, * rval;
    UserData data;
    realtype* x_cells, * T_vect, * Y_vect, * Tp_vect, * Yp_vect;
    double M;
    data = (UserData)user_data;
    T_vect = data->T;
    Y_vect = data->Y_H2O;
    x_cells = data->x;
    int j;
    yval = N_VGetArrayPointer(yy);
    ypval = N_VGetArrayPointer(yp);
    rval = N_VGetArrayPointer(rr);
    //cout << "ypvalres0 = " << yval[0] << "\n";
    int myNx = data->Nx;
    ExportToArray(T_vect, Y_vect, data->M, data, yy, data->Nx);
    int k = 0;
    double Y_H2, Y_O2;
    double W;
    double rho;
    for (j = 1; j < myNx - 1; j++) {
        if (j != data->N_centr)
        {
            get_Y(Y_vect[j], Y_H2, Y_O2, Y_N2);
            W = get_W(Y_vect[j], Y_H2, Y_O2, Y_N2);
            //cout << "W = " << W << "\n";
            rho = P * W / phyc.kR / T_vect[j] * pow(10, -3);
            rval[j - 1] = F_right(T_vect[j - 1], T_vect[j], T_vect[j + 1], data->M, Y_vect[j], Y_vect[j + 1], x_cells, j) / rho + ypval[k];
            k++;
        }
        else
        {
            rval[j - 1] = F_right(T_vect[j - 1], T_vect[j], T_vect[j + 1], data->M, Y_vect[j], Y_vect[j + 1], x_cells, j);
        }
        //cout << "right  = " << j - 1 << "  k = " << k - 1 << "\n";
    }
    //cout << "next\n";
    //cout << "N_m = " << data->N_m << "\n";
    //cout << "M = " << yval[data->N_m] << "\n";
    for (int i = 1; i < myNx - 1; i++) {
        get_Y(Y_vect[i], Y_H2, Y_O2, Y_N2);
        W = get_W(Y_vect[i], Y_H2, Y_O2, Y_N2);
        //cout << "W = " << W << "\n";
        rho = P * W / phyc.kR / T_vect[i] * pow(10, -3);
        rval[j - 1] = F_rightY(T_vect[i - 1], T_vect[i], T_vect[i + 1], data->M, Y_vect[i - 1], Y_vect[i], Y_vect[i + 1], x_cells, i) / rho
            + ypval[j - 1];
        //cout << "right  = " << j - 1 << "  k = " << j - 1 << "\n";
        j++;
    }
    return(0);
}

static int check_retval(void* retvalvalue, const char* funcname, int opt)
{
    int* errretval;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && retvalvalue == NULL) {
        fprintf(stderr,
            "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
        return(1);
    }

    /* Check if retval < 0 */
    else if (opt == 1) {
        errretval = (int*)retvalvalue;
        if (*errretval < 0) {
            fprintf(stderr,
                "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
                funcname, *errretval);
            return(1);
        }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && retvalvalue == NULL) {
        fprintf(stderr,
            "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
        return(1);
    }

    return(0);
}
// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.

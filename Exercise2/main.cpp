#include <iostream>
#include <math.h>
#include <Eigen/Eigen>
#include <iomanip>

using namespace std;
using namespace Eigen;

bool SolveSystemPALU(const MatrixXd& A,
                 const VectorXd& b,
                 double& errRel)
{
    JacobiSVD<MatrixXd> svd(A);
    VectorXd singularValuesA = svd.singularValues();

    if( singularValuesA.minCoeff() < 1e-16)
    {
        errRel = -1;
        return false;
    }

    unsigned int n = A.rows();
    VectorXd exactSolution = -VectorXd::Ones(n);
    VectorXd x = A.fullPivLu().solve(b);

    errRel = (exactSolution - x).norm() / exactSolution.norm();

    return true;
}

bool SolveSystemQR(const MatrixXd& A,
                     const VectorXd& b,
                     double& errRel)
{
    JacobiSVD<MatrixXd> svd(A);
    VectorXd singularValuesA = svd.singularValues();

    if( singularValuesA.minCoeff() < 1e-16)
    {
        errRel = -1;
        return false;
    }

    unsigned int n = A.rows();
    VectorXd exactSolution = -VectorXd::Ones(n);
    VectorXd x = A.householderQr().solve(b);

    errRel = (exactSolution - x).norm() / exactSolution.norm();

    return true;
}

int main()
{

    Matrix2d A1{
        {5.547001962252291e-01, -3.770900990025203e-02},
        {8.320502943378437e-01, -9.992887623566787e-01}
    };
    Vector2d b1{
        {-5.169911863249772e-01},
        {1.672384680188350e-01}
    };
    double errRel1, errRel1_qr;
    if(SolveSystemPALU(A1, b1, errRel1))
        cout << scientific << setprecision(16) << "A1: Relative Error using PALU decomposition: "<< errRel1<< endl;
    else
        cout <<"Matrix A1 is singular" << endl;
    if(SolveSystemQR(A1, b1, errRel1_qr))
        cout << scientific << "A1: Relative Error using QR decomposition: "<< errRel1_qr << endl;
    else
        cout <<"Matrix A1 is singular" << endl;


    Matrix2d A2{
        {5.547001962252291e-01, -5.540607316466765e-01},
        {8.320502943378437e-01, -8.324762492991313e-01}
    };
    Vector2d b2{
        {-6.394645785530173e-04},
        {4.259549612877223e-04}
    };
    double errRel2, errRel2_qr;
    if(SolveSystemPALU(A2, b2, errRel2))
        cout << scientific<< "A2: Relative Error using PALU decomposition: "<< errRel2<< endl;
    else
        cout <<" Matrix A2 is singular)"<< endl;
    if(SolveSystemQR(A2, b2, errRel2_qr))
        cout << scientific << "A2: Relative Error using QR decomposition: "<< errRel2_qr << endl;
    else
        cout << "Matrix A2 is singular" << endl;


    Matrix2d A3{
        {5.547001962252291e-01, -5.547001955851905e-01},
        {8.320502943378437e-01, -8.320502947645361e-01}
    };
    Vector2d b3{
        {-6.400391328043042e-10},
        {4.266924591433963e-10}
    };
    double errRel3, errRel3_qr;
    if(SolveSystemPALU(A3, b3, errRel3))
        cout << scientific<< "A3: Relative Error using PALU decomposition: "<< errRel3<< endl;
    else
        cout << "Matrix A3 is singular"<< endl;
    if(SolveSystemQR(A3, b3, errRel3_qr))
        cout << scientific << "A3: Relative Error using QR decomposition: "<< errRel3_qr << endl;
    else
        cout << "Matrix A3 is singular" << endl;

    return 0;

}

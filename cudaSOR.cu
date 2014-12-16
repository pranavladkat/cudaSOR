#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <ctime>

using namespace std;

// global variables
size_t const BLOCK_SIZE = 16;
size_t const width = 8;
size_t const height = 8;
size_t itmax = 3000;
double const omega = 1.97;
double const beta = (((1.0/width) / (1.0/height))*((1.0/width) / (1.0/height)));


// functions
void generategrid(double*,double*,const double,const double,const double,const double);

void setBC(double*, const double*, const double*);
void solve_sor_cuda(double*);
__global__ void solve_odd(double*,double*);
__global__ void solve_even(double*,double*);
__global__ void merge_oddeven(double*,double*);
void host_sor(double*,double*);


// boundary conditions
double leftBC(const double&, const double&);
double rightBC(const double&, const double&);
double topBC(const double&, const double&);
double bottomBC(const double&, const double&);

int main(){

  //host variables
  double *x, *y;        // grid x and y
  double Xmin = 0.0, Xmax = 1.0,
         Ymin = 0.0, Ymax = 1.0;    // grid coordinates bounds
  double *sol;

  // allocate memory for grid
  size_t memsize = width*height;
  x = new double [memsize];
  y = new double [memsize];

  // generate grid
  generategrid(x,y,Xmin,Xmax,Ymin,Ymax);

  // allocate sol memory + set it to zero
  sol = new double [memsize];
  memset(sol,memsize,memsize*sizeof(double));

  // set boundary conditions
  setBC(sol,x,y);


  solve_sor_cuda(sol);


  ofstream file("cusol.dat");
  for(int i = 0; i < width; i++){
    for(int j = 0; j < height; j++){
      file << setw(12) << sol[i*height + j];
    }
    file << endl;
  }
  file.close();

  delete [] x;
  delete [] y;
  delete [] sol;

  cout << "End!" << endl;
  return 0;
}



void generategrid(double* x,double* y,const double Xmin,const double Xmax,const double Ymin,const double Ymax){

  double dx = fabs(Xmax-Xmin)/(width-1);
  double dy = fabs(Ymax-Ymin)/(height-1);

  for(size_t i = 0; i < width; i++){
      for(size_t j = 0; j < height; j++){
          x[i*height + j] = Xmin + i*dx;
          y[i*height + j] = Ymin + j*dy;
          //cout << setw(12) << y[i*height + j];
      }
      //cout << endl;
  }
}


void setBC(double* sol,const double* x, const double* y){

  for(size_t i = 0; i < width; i++){
      for(size_t j = 0; j < height; j++){

          size_t index = i*height + j ;
          if(i == 0){
              sol[index] = leftBC(x[index],y[index]);
          }
          if(i == width-1){
              sol[index] = rightBC(x[index],y[index]);
          }
          if(j == 0){
              sol[index] = bottomBC(x[index],y[index]);
          }
          if(j == height-1){
              sol[index] = topBC(x[index],y[index]);
          }
      }
  }
}


// boundary conditions
double leftBC(const double& x, const double& y){
  return 0;
}

double rightBC(const double& x, const double& y){
  return 0;
}

double topBC(const double& x, const double& y){
  return sin(M_PI*x)*exp(-M_PI);
}

double bottomBC(const double& x, const double& y){
  return sin(M_PI*x);
}



void solve_sor_cuda(double* sol_host){

  int nGPU;
  int mywidth = width/2;
  size_t memsize = mywidth*height;

  // get device count
  cudaGetDeviceCount(&nGPU);
  cout << nGPU << " cuda devices found." << endl;

  double *odd, *even;

  odd = new double [memsize];
  even = new double [memsize];

  int p = 0;
  for(size_t i = 0; i < width; i++){
      for(size_t j = 0; j < height; j++){
          size_t index = i*height+j;
          sol_host[index] = p;
          p++;
      }
  }


  for(size_t i = 0; i < mywidth; i++){
      for(size_t j = 0; j < height; j++){
          size_t index = i*height+j;
          odd[index] = sol_host[index];
          //even[index] = sol_host[index+]
          cout << setw(10) << odd[index] ;
      }
      cout << endl;
  }



}





















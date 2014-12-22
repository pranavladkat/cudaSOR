#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <omp.h>


using namespace std;

// global variables
size_t const BLOCK_SIZE = 16;
size_t const width = 800;
size_t const height = 800;
size_t itmax = 5000;
double const omega = 1.97;
double const beta = (((1.0/width) / (1.0/height))*((1.0/width) / (1.0/height)));


// functions
void generategrid(double*,double*,const double,const double,const double,const double);
void setBC(double*, const double*, const double*);
void solve_sor_host(double*);
void solve_sor_cuda(double*);
__global__ void solve_odd(double*,double*);
__global__ void solve_even(double*,double*);
__global__ void merge_oddeven(double*,double*);
void write_output(double*);


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
  memset(sol,0,memsize*sizeof(double));

  // set boundary conditions
  setBC(sol,x,y);

  // call solvers
  //solve_sor_cuda(sol);
  solve_sor_host(sol);

  // write output
  write_output(sol);

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


  const int memsize = width*height;

  // device variables -> odd and even
  double *odd, *even;

  cudaMalloc(&odd,memsize*sizeof(double));
  cudaMalloc(&even,memsize*sizeof(double));
  cudaMemset(&odd,0,memsize);
  cudaMemset(&even,0,memsize);

  // copy initial guess from host to device memory
  cudaMemcpy(odd,sol_host,memsize*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(even,sol_host,memsize*sizeof(double),cudaMemcpyHostToDevice);

  int gridx = (width-1)/BLOCK_SIZE + 1;
  int gridy = (height-1)/BLOCK_SIZE + 1;
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
  dim3 dimGrid(gridx,gridy,1);

  cout << "gridx = " << gridx << "\t" << "gridy = " << gridy << endl;

  double stime = clock();
  for(size_t it = 0; it < itmax; it++){
      solve_odd <<<dimGrid,dimBlock>>> (odd,even);
      solve_even <<<dimGrid,dimBlock>>> (odd,even);
  }
  merge_oddeven <<<dimGrid,dimBlock>>> (odd,even);
  double etime = clock();
  cout << "GPU time : " << (etime-stime)/CLOCKS_PER_SEC << endl;

  // copy solution from device to host memory
  cudaMemcpy(sol_host,odd,memsize*sizeof(double),cudaMemcpyDeviceToHost);

  cudaFree(odd);
  cudaFree(even);

}



__global__ void solve_odd(double* odd,double* even){

  size_t tx = blockIdx.x*blockDim.x + threadIdx.x;
  size_t ty = blockIdx.y*blockDim.y + threadIdx.y;
  size_t index = tx*height+ty;

  if((tx + ty)%2 != 0){
      if(tx > 0 && ty > 0 && tx < width-1 && ty < height-1){
          odd[index] = (1.0-omega)*odd[index] + omega/(2*(1+beta))
                     *(even[index+1] + even[index-1] + beta*(even[index+height] + even[index-height]));
      }
  }
}


__global__ void solve_even(double* odd,double* even){

  size_t tx = blockIdx.x*blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y*blockDim.y + threadIdx.y;
    size_t index = tx*height+ty;
    //double beta = pow((1.0/width) / (1.0/height),2);

    //even
    if((tx + ty)%2 == 0){
        if(tx > 0 && ty > 0 && tx < width-1 && ty < height-1){
            even[index] = (1.0-omega)*even[index] + omega/(2*(1+beta))
                        *(odd[index+1] + odd[index-1] + beta*(odd[index+height] + odd[index-height]));
        }
    }


}



__global__ void merge_oddeven(double* odd,double* even){

  size_t tx = blockIdx.x*blockDim.x + threadIdx.x;
  size_t ty = blockIdx.y*blockDim.y + threadIdx.y;
  size_t index = tx*height+ty;

  if((tx + ty)%2 == 0 && tx < width-1 && ty < height-1){
      odd[index] = even[index];
  }

}



void solve_sor_host(double* sol){

  const int memsize = width*height;
  double *odd, *even;
  size_t i,j, index;

  odd = new double [memsize];
  even = new double [memsize];

  memcpy(odd,sol,memsize*sizeof(double));
  memcpy(even,sol,memsize*sizeof(double));


  double stime = omp_get_wtime();
  for(size_t it = 0; it < itmax; it++){

#pragma omp parallel
{
#pragma omp for private(i,j,index)
      // update odd cells
      for(i = 0; i < width; i++){
          for(j = 0; j < height; j++){
              index = i*height+j;
              if((i + j)%2 != 0){
                  if(i > 0 && j > 0 && i < width-1 && j < height-1){
                      odd[index] = (1.0-omega)*odd[index] + omega/(2*(1+beta))
                                 *(even[index+1] + even[index-1] + beta*(even[index+height] + even[index-height]));
                  }
              }
          }
      }

#pragma omp for private(i,j,index)
      // update even cells
      for(i = 0; i < width; i++){
          for(j = 0; j < height; j++){
              index = i*height+j;
              if((i + j)%2 == 0){
                  if(i > 0 && j > 0 && i < width-1 && j < height-1){
                      even[index] = (1.0-omega)*even[index] + omega/(2*(1+beta))
                                  *(odd[index+1] + odd[index-1] + beta*(odd[index+height] + odd[index-height]));
                  }
              }
          }
      }
} // end omp parallel

  } // end iteration loop

#pragma omp parallel for private(i,j,index)
  // merge odd-even solution
  for(i = 0; i < width; i++){
      for(j = 0; j < height; j++){
          index = i*height+j;
          if((i + j)%2 == 0){
              odd[index] = even[index];
          }
      }
  }
  double etime = omp_get_wtime();
  cout << "CPU time : " << (etime-stime) << endl;


  // copy solution from odd to sol
  memcpy(sol,odd,memsize*sizeof(double));

  delete [] odd;
  delete [] even;

}


void write_output(double* sol){

  ofstream file("cusol.dat");
    for(int i = 0; i < width; i++){
      for(int j = 0; j < height; j++){
        file << setw(12) << sol[i*height + j];
      }
      file << endl;
    }
    file.close();
}



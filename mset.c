#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "mpi.h"

#define ITER 256
#define WIDTH 1200
#define HEIGHT 1200
#define RMIN -2.5
#define RMAX 1.5
#define IMIN -2.0
#define IMAX 2.0

struct complex {
    double real;
    double imag;
};

int getIterations(struct complex c) {
    int count, max_iter;
    struct complex z;
    double temp = 0;
    double lengthsq = 0;
    z.real = 0;
    z.imag = 0;
    count = 0;
    while ((lengthsq<=4.0) && (count<ITER)) {
        temp = z.real*z.real - z.imag*z.imag + c.real;
        z.imag = 2*z.real*z.imag + c.imag;
        z.real = temp;
        lengthsq = z.real*z.real + z.imag*z.imag;
        count++;
    }
    return count;
}

void sequential() {
    struct complex c;
    double scale_real = (RMAX - RMIN)/WIDTH;
    double scale_imag = (IMAX - IMIN)/HEIGHT;
    FILE *image;
    image = fopen("Mandelbrot.ppm","w");
    fprintf(image, "P3\n%d %d\n255\n", WIDTH, HEIGHT);
    for (int y=HEIGHT-1; y>0; y--) {
        for (int x=0; x<WIDTH; x++) {
            c.real = RMIN + ((double)x*scale_real);
            c.imag = IMIN + ((double)y*scale_imag);
            int iters = getIterations(c);
            int color = iters;
            printf("Coloring pixel x: %d y: %d color: %d.     \r",x, y, color);
            fprintf(image, "%d %d %d ",color, color, color);
        }
    }
    fclose(image);
}

double parastatic(int rank, int size) {
    if (rank==0) {
        double masterCommTime;
        char processor[MPI_MAX_PROCESSOR_NAME];
        int namelen;
        MPI_Get_processor_name(processor, &namelen);
        MPI_Status status;
    	printf("Master Rank = %d on %s: Sending rows...\n",rank,processor);
        MPI_Barrier(MPI_COMM_WORLD);
        int row = 0;
        int increment = HEIGHT/(size-1);
        for (int i=1; i<size; i++) {
            MPI_Send(&row, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            row+=increment;
        }
        
        printf("Master Rank = %d on %s: Rows sent.\n",rank,processor);
        
        printf("Master Rank = %d on %s: Receiving rows...\n",rank,processor);
        MPI_Barrier(MPI_COMM_WORLD);
        int color = 0;
        int x = 0, y = 0;
        int ** cols = malloc(WIDTH*sizeof(int *));
        for (int i=0; i<WIDTH; i++) {
            cols[i] = malloc(HEIGHT*sizeof(int));
        }
        for (int i=0; i<WIDTH; i++) {
            for (int j=0; j<HEIGHT; j++) {
                cols[i][j] = 0;
            }
        }
        int* pixel = malloc(sizeof(int)*3);
        for (int i=0; i<(WIDTH*HEIGHT); i++) {
            MPI_Recv(pixel, 3, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            cols[pixel[1]][pixel[2]] = pixel[0]; 
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        double maxCompTime = 0.0;
        double temp = 0.0;
        for (int i=0; i<size-1; i++) {
            MPI_Recv(&temp, sizeof(double), MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            if (temp > maxCompTime) {
                maxCompTime = temp;
            }
        }

        printf("Master Rank = %d on %s: Received rows.\n",rank,processor);
        
        clock_t colorStart, colorEnd;
        colorStart = clock();
        printf("Master Rank = %d on %s: Coloring image...\n",rank,processor);
        FILE *image;
        image = fopen("Mandelbrot.ppm","w");
        fprintf(image, "P3\n%d %d\n255\n", WIDTH, HEIGHT);
        int temp_col = 0;
        for (int i=0; i<WIDTH; i++) {
            for (int j=0; j<HEIGHT; j++) {
                temp_col = cols[j][i];
                fprintf(image, "%d %d %d ",temp_col, temp_col, temp_col);
            }
        }
        fclose(image);
        colorEnd = clock();
        maxCompTime += ((double)(colorEnd - colorStart))/CLOCKS_PER_SEC;
        printf("Master Rank = %d on %s: Image colored.",rank,processor);
        return maxCompTime;
    }
    else {
        char processor[MPI_MAX_PROCESSOR_NAME];
        int namelen;
        MPI_Get_processor_name(processor, &namelen);
        MPI_Status status;
        MPI_Barrier(MPI_COMM_WORLD);
        printf("Slave Rank = %d on %s: Receiving rows...\n",rank,processor);
        int row = 0;
        MPI_Recv(&row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Slave Rank = %d on %s: Received rows %d.\n",rank,processor,row);

        
        printf("Slave Rank = %d on %s: Sending colors and coords...\n",rank,processor);
        MPI_Barrier(MPI_COMM_WORLD);
        clock_t computationStart;
        clock_t computationEnd;
        double computationTime = 0.0;

        int increment = HEIGHT/(size-1);
        struct complex c;
        double scale_real = (RMAX - RMIN)/WIDTH;
        double scale_imag = (IMAX - IMIN)/HEIGHT;
        int x = 0, y = 0;
        int source = rank;
        int color = 0;
        int* pixel = malloc(sizeof(int)*3);

        for (x=0; x<WIDTH; x++) {
            for (y=row; y<(row+increment); y++) {
                computationStart = clock();
                c.real = RMIN + ((double)x*scale_real);
                c.imag = IMIN + ((double)y*scale_imag);
                int iters = getIterations(c);
                pixel[0] = iters;
                pixel[1] = x;
                pixel[2] = y;
                computationEnd = clock();
                computationTime += ((double)(computationEnd - computationStart))/CLOCKS_PER_SEC;
                MPI_Send(pixel, 3, MPI_INT, 0, 0, MPI_COMM_WORLD);
                printf("Slave Rank = %d on %s: Sending pixel x: %d y: %d color: %d.       \r\r",rank, processor, pixel[1], pixel[2], pixel[0]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        MPI_Send(&computationTime, sizeof(MPI_DOUBLE), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

        printf("Slave Rank = %d on %s: Colors and coords sent.\n",rank,processor);
        return 0.0;
    }
}

double paradynamic(int rank, int size) {
    if (rank==0) {
        int count = 0;
        int row = 0;
        for (int k=0; k<(size-1); k++) {
            MPI_Send(&row, sizeof(MPI_INT), MPI_INT, k, 0, MPI_COMM_WORLD);
            count++;
            row++;
        }

        int ** cols = malloc(WIDTH*sizeof(int *));
        for (int i=0; i<WIDTH; i++) {
            cols[i] = malloc(HEIGHT*sizeof(int));
        }
    }
    else {
        //
    }
}

void main(int argc, char **argv) {
    char choice = (char)*argv[1];
    MPI_Init(0,0);
    
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    clock_t start, end;
    double compTime = 0.0;
    if (choice=='p') {
        if (HEIGHT%(size-1)!=0) {
            printf("Bad processor count");
            exit(1);
        }
    }
    if (rank==0) {
        printf("\nStarting...\n\n");
        start = clock();
    }
    if (choice=='p') {
        compTime = parastatic(rank,size);
    }
    else {
        if (choice=='s') {
        sequential();
        }
        else {
            exit(1);
        }
    }

    if (rank==0) {
        end = clock();
        double Tcputime = (double)(end-start)/CLOCKS_PER_SEC;
        
        printf("\nDone!\n");
        printf("Time taken: %f\n",Tcputime);
        
        if (choice=='p') {
            printf("Computation Time: %f\n",compTime);
            printf("Communcation Time: %f\n",Tcputime-compTime);
        }
    }
    MPI_Finalize();
}

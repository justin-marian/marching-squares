#include "helpers.h"
 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
 
#include <pthread.h>
 
#define RGB_SIZE                3
 
#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
 
#define SIGMA                   200
 
#define STEP_X                  8
#define STEP_Y                  8

#define RESCALE_X               2048
#define RESCALE_Y               2048
 
#define CLAMP(v, min, max) \
    if(v < min)            \
    { v = min; }           \
    else if(v > max)       \
    { v = max; }
 
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

typedef struct PPM_Thread_Info {
    int Thread_ID;
    int Thread_Count;

    pthread_barrier_t *Barrier;
    
    int p, q;
    int Start_Scaled, End_Scaled;

    ppm_image *Initial_Image;
    ppm_image **Contour_Map;
    ppm_image *Scaled_Image;

    uint8_t **Grid_Image;
} PPM_Thread;


/**
 * @brief Initializes a map between binary configurations and contour images.
 *
 * Creates a map where binary configurations (e.g., 0110_2) are
 * associated with the corresponding contour images. The map is an array, and
 * each index corresponds to a specific configuration.
 *
 * @return An array of pointers to ppm_image representing the contour map.
 */
ppm_image **init_contour_map(void) {
    ppm_image **Map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
 
    if (!Map) {
        fprintf(stderr, "ERROR: ALLOCATE CONTOUR MAP...\n");
        exit(EXIT_FAILURE);
    }
 
    int i;
 
    for (i = 0; i < CONTOUR_CONFIG_COUNT; ++i) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        Map[i] = read_ppm(filename);
    }
 
    return Map;
}


/**
 * @brief Frees allocated resources.
 *
 * Deallocates memory for contour maps, grid images, and image data.
 *
 * @param Image The original image.
 * @param Contour_Map An array of pointers to ppm_image representing the contour map.
 * @param Grid_Image The grid image represented as a 2D array.
 * @param StepX The step size for rescaling.
 */
void free_resources(ppm_image *Image, ppm_image **Contour_Map, uint8_t **Grid_Image, int StepX) {
    int i;
 
    for (i = 0; i < CONTOUR_CONFIG_COUNT; ++i) {
        free(Contour_Map[i]->data);
        free(Contour_Map[i]);
    }
    free(Contour_Map);
 
    for (i = 0; i <= Image->x / StepX; ++i) {
        free(Grid_Image[i]);
    }
    free(Grid_Image);
 
    free(Image->data);
    free(Image);
}


/**
 * @brief Updates a section of an image with contour pixels.
 *
 * Copies pixel values from a contour image to a section of the
 * original image starting at coordinates (x, y).
 *
 * @param Image The original image.
 * @param Contour The contour image to copy from.
 * @param x The x-coordinate to start copying.
 * @param y The y-coordinate to start copying.
 */
void update_image(ppm_image *Image, ppm_image *Contour, int x, int y) {
    int i, j;
 
    for (i = 0; i < Contour->x; ++i) {
        for (j = 0; j < Contour->y; ++j) {
            int Contour_Pixel_Index = Contour->x * i + j;
            int Image_Pixel_Index = (x + i) * Image->y + y + j;
 
            Image->data[Image_Pixel_Index].red = Contour->data[Contour_Pixel_Index].red;
            Image->data[Image_Pixel_Index].green = Contour->data[Contour_Pixel_Index].green;
            Image->data[Image_Pixel_Index].blue = Contour->data[Contour_Pixel_Index].blue;
        }
    }
}


/**
 * @brief Rescales the given image to a predefined size.
 * 
 * Take the original image provided in the thread information
 * and rescale it to the dimensions specified by RESCALE_X and RESCALE_Y using
 * bicubic sampling. It will only perform rescaling if the original image's dimensions
 * are greater than the desired scale dimensions.
 * 
 * @param PPM_Infos Pointer to a structure containing thread information and image data.
 */
void rescale_image(PPM_Thread* PPM_Infos) {
    int i, j;
    uint8_t RGB[RGB_SIZE];

    if (PPM_Infos->Initial_Image->x <= RESCALE_X && 
        PPM_Infos->Initial_Image->y <= RESCALE_Y) return;

    PPM_Infos->Start_Scaled = PPM_Infos->Thread_ID * (double)RESCALE_X / PPM_Infos->Thread_Count;
    PPM_Infos->End_Scaled = MIN((PPM_Infos->Thread_ID + 1) * (double)RESCALE_X / PPM_Infos->Thread_Count, RESCALE_X);
    
    for (i = PPM_Infos->Start_Scaled; i < PPM_Infos->End_Scaled; ++i) {
        for (j = 0; j < PPM_Infos->Scaled_Image->y; ++j) {
            float u = (float)i / (float)(PPM_Infos->Scaled_Image->x - 1);
            float v = (float)j / (float)(PPM_Infos->Scaled_Image->y - 1);

            sample_bicubic(PPM_Infos->Initial_Image, u, v, RGB);

            int Scaled_Idx = i * PPM_Infos->Scaled_Image->y + j;
            PPM_Infos->Scaled_Image->data[Scaled_Idx].red = RGB[0];
            PPM_Infos->Scaled_Image->data[Scaled_Idx].green = RGB[1];
            PPM_Infos->Scaled_Image->data[Scaled_Idx].blue = RGB[2];
        }
    }

    pthread_barrier_wait(PPM_Infos->Barrier);
}


/**
 * @brief Samples the grid for the marching squares algorithm.
 * 
 * Iterate over the grid cells corresponding to the thread's assigned
 * portion of the image and assigns binary values based on the average color of the
 * pixels and the sigma threshold. It also handles edge cases for the rightmost column
 * and the bottom row if the current thread is the last one (master thread).
 * 
 * @param PPM_Infos Pointer to a structure containing thread information and image data.
 */
void sample_grid(PPM_Thread* PPM_Infos) {
    int i, j;

    int start = PPM_Infos->Start_Scaled = PPM_Infos->Thread_ID * (double) PPM_Infos->p / PPM_Infos->Thread_Count;
    int end = PPM_Infos->End_Scaled = MIN((PPM_Infos->Thread_ID + 1) * (double) PPM_Infos->p / PPM_Infos->Thread_Count, PPM_Infos->p);

    for (i = start; i < end; ++i) {
        for (j = 0; j < PPM_Infos->q; ++j) {
            ppm_pixel Curr_Pixel = PPM_Infos->Scaled_Image->data[i * STEP * PPM_Infos->Scaled_Image->y + j * STEP];
            uint8_t Curr_Color = (Curr_Pixel.red + Curr_Pixel.green + Curr_Pixel.blue) / RGB_SIZE;
            PPM_Infos->Grid_Image[i][j] = (Curr_Color > SIGMA) ? 0 : 1;
        }
    }

    for (i = start; i < end; ++i) {
        ppm_pixel Curr_Pixel = PPM_Infos->Scaled_Image->data[i * STEP * PPM_Infos->Scaled_Image->y + PPM_Infos->Scaled_Image->x - 1];
        uint8_t Curr_Color = (Curr_Pixel.red + Curr_Pixel.green + Curr_Pixel.blue) / RGB_SIZE;
        PPM_Infos->Grid_Image[i][PPM_Infos->q] = (Curr_Color > SIGMA) ? 0 : 1;
    }

    int MASTER = PPM_Infos->Thread_Count - 1;
    if (PPM_Infos->Thread_ID == MASTER) {
        for (j = 0; j < PPM_Infos->q; ++j) {
            ppm_pixel Curr_Pixel = PPM_Infos->Scaled_Image->data[(PPM_Infos->Scaled_Image->x - 1) * PPM_Infos->Scaled_Image->y + j * STEP];
            uint8_t Curr_Color = (Curr_Pixel.red + Curr_Pixel.green + Curr_Pixel.blue) / RGB_SIZE;
            PPM_Infos->Grid_Image[PPM_Infos->p][j] = (Curr_Color > SIGMA) ? 0 : 1;
        }
    }

    pthread_barrier_wait(PPM_Infos->Barrier);
}


/**
 * @brief Applies the marching squares algorithm to generate contours.
 * 
 * Apply the marching squares algorithm on the sampled grid to generate contours.
 * It updates the Scaled_Image with the corresponding contour images. The function uses
 * the Contour_Map member of the PPM_Infos structure to map grid patterns to contour images.
 * 
 * @param PPM_Infos Pointer to a structure containing thread information, image data, and grid data.
 */
void march(PPM_Thread* PPM_Infos) {
    int i, j;

    int start = PPM_Infos->Start_Scaled;
    int end = PPM_Infos->End_Scaled;

    for (i = start; i < end; ++i) {
        for (j = 0; j < PPM_Infos->q; ++j) {
           uint8_t k = 8 * PPM_Infos->Grid_Image[i][j] + 
                       4 * PPM_Infos->Grid_Image[i][j + 1] +
                       2 * PPM_Infos->Grid_Image[i + 1][j + 1] +
                       1 * PPM_Infos->Grid_Image[i + 1][j];
            update_image(PPM_Infos->Scaled_Image,
                         PPM_Infos->Contour_Map[k],
                         i * STEP, j * STEP);
        }
    }

    pthread_barrier_wait(PPM_Infos->Barrier);
}


/**
 * @brief Orchestrate the rescaling, grid sampling, and marching squares.
 * 
 * Starting point for each thread created by the main function.
 * It calls the rescale_image, sample_grid, and march_grid functions in sequence
 * to perform the multithreaded rescaling and contour mapping of the image.
 * 
 * @param arg A void pointer to the thread-specific information structure.
 * @return NULL after completing the execution.
 */
void *rescale_grid_march(void *arg) {
    PPM_Thread* PPM_Infos = (PPM_Thread*) arg;
    rescale_image(PPM_Infos); // 1
    sample_grid(PPM_Infos); // 2
    march(PPM_Infos); // 3
    pthread_exit(NULL);
}


int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "USAGE: ./tema1_par <IN_FILE> <OUT_FILE> <P>\n");
        return EXIT_FAILURE;
    }

    ppm_image *Image = read_ppm(argv[1]);
    int P = atoi(argv[3]);
    int p = Image->x / STEP_X;
    int q = Image->y / STEP_Y;
    ppm_image *Scaled = (ppm_image *)malloc(sizeof(ppm_image));

    if (!Scaled) {
        fprintf(stderr, "UNABELE TO ALLOCATE MEMORY FOR SCALED IMAGE\n");
        exit(EXIT_FAILURE);
    }

    Scaled->y = RESCALE_Y;
    Scaled->x = RESCALE_X;
    Scaled->data = (ppm_pixel *)malloc(Scaled->x * Scaled->y * sizeof(ppm_pixel));
 
    if (!Scaled->data) {
        fprintf(stderr, "UNABLE TO ALLOCATE MEMORY FOR SCALED IMAGE DATA\n");
        exit(EXIT_FAILURE);
    }

    if (Image->x > RESCALE_X && Image->y > RESCALE_Y) {
        p = Scaled->x / STEP_X;
        q = Scaled->y / STEP_Y;
    }
  
    uint8_t **Grid = (uint8_t **)malloc((p + 1) * sizeof(uint8_t*));

    if (!Grid) {
        fprintf(stderr, "UNABLE TO ALLOCATE MEMORY FOR GRID\n");
        exit(EXIT_FAILURE);
    }
 
    for (int i = 0; i <= p; ++i) {
        Grid[i] = (uint8_t *)malloc((q + 1) * sizeof(uint8_t));
 
        if (!Grid[i]) {
            fprintf(stderr, "UNABLE TO ALLOCATE MEMORY FOR GRID ROWS\n");
            // Free previously allocated memory
            for (int j = 0; j < i; ++j) {
                free(Grid[j]);
            }
            free(Grid);
            
            free(Scaled->data);
            free(Scaled);

            exit(EXIT_FAILURE);
        }
    }

    // 0. Initialize contour map
    ppm_image **Contour = init_contour_map();

    int id, rc;
    void *status;

    pthread_t Threads[P];
    PPM_Thread PPM_Infos[P];

    pthread_barrier_t Barrier;

    pthread_barrier_init(&Barrier, NULL, P);
    // Split the work between threads.
    for (id = 0; id < P; ++id) {
        PPM_Infos[id].Thread_Count = P;
        PPM_Infos[id].Thread_ID = id;
        
        PPM_Infos[id].p = p;
        PPM_Infos[id].q = q;
        PPM_Infos[id].Barrier = &Barrier;

        PPM_Infos[id].Contour_Map = Contour;
        PPM_Infos[id].Initial_Image = Image;
        PPM_Infos[id].Scaled_Image = Image;
        PPM_Infos[id].Grid_Image = Grid;
        
        if (Image->y > RESCALE_Y && Image->x > RESCALE_X) PPM_Infos[id].Scaled_Image = Scaled;

        PPM_Infos[id].End_Scaled = MIN((PPM_Infos[id].Thread_ID + 1) * (double)RESCALE_X / PPM_Infos[id].Thread_Count, RESCALE_X);
        PPM_Infos[id].Start_Scaled = PPM_Infos[id].Thread_ID * (double)RESCALE_X / PPM_Infos[id].Thread_Count;

        rc = pthread_create(&Threads[id], NULL, rescale_grid_march, &PPM_Infos[id]);
 
        if (rc) {
            fprintf(stderr, "ERROR: CREATE THREAD ID: %d", id);
            exit(EXIT_FAILURE);
        }
    }
  	// Wait for all threads to finish.
    for (id = 0; id < P; ++id) {
        rc = pthread_join(Threads[id], &status);
 
        if (rc) {
            fprintf(stderr, "ERROR: WAITING THREAD ID: %d", id);
            exit(EXIT_FAILURE);
        }
    }
    pthread_barrier_destroy(&Barrier);

    // 4. Write output
    write_ppm(PPM_Infos[0].Scaled_Image, argv[2]);

    free_resources(Scaled, Contour, Grid, STEP);

    return EXIT_SUCCESS;
}

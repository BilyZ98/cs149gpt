#include <cmath>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
  int z_count = z * sizeZ;
  int y_count = y * sizeY * sizeZ;
  int x_count = x * sizeX * sizeY * sizeZ;

    return tensor[x_count + y_count + z_count + b];
    return 0.0;
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {

  int z_count = z * sizeZ;
  int y_count = y * sizeY * sizeZ;
  int x_count = x * sizeX * sizeY * sizeZ;

  tensor[x_count + y_count + z_count + b] = val;
    return; 
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; b++) {
       //loop over Heads
       for (int h = 0; h < H; h++) {
          //loop over Sequence Length
          for (int i = 0; i < N; i++) {
            for(int seq_i=0; seq_i < N; seq_i++) {
             //loop over Embedding Dimensionality
              float val = 0.0;
               for (int j = 0; j < d; j++) {
                  int q_row  = i; 
                  int q_col = j;
                  int k_row = seq_i;
                  int k_col = j;
                  // float val = fourDimRead(Q, b, h, i, j, H, N, d);
                  float q_val = fourDimRead(Q, b, h, q_row, q_col, H, N, d);
                  float k_val = fourDimRead(K, b, h, k_row, k_col, H, N, d);
                  val += q_val * k_val;
               }
              twoDimWrite(QK_t, i, seq_i, N, val );
            }

           }
            for(int row_idx=0; row_idx < N; row_idx++) {
              std::vector<double> tmp_row_res(N, 0.0);
              double row_sum = 0.0;
              for(int cold_idx=0; cold_idx < N ;cold_idx++) {
                 float val = twoDimRead(QK_t, row_idx, cold_idx, N);
                double exp_val = std::exp(val);
                row_sum += exp_val;
                tmp_row_res[cold_idx] = exp_val;
              }
              for(int cold_idx=0; cold_idx < N ; cold_idx++) {
                float prob = tmp_row_res[cold_idx] / row_sum;
                twoDimWrite(QK_t, row_idx, cold_idx, N, prob);
              }
            }


            for(int qkt_row_idx=0; qkt_row_idx < N; qkt_row_idx++) {
            for(int output_d_idx=0; output_d_idx < d; output_d_idx++) {
              float val =0.0;
              for(int m_idx=0; m_idx < N ; m_idx++) {
                float qkt_val =  twoDimRead(QK_t, qkt_row_idx, m_idx, N);
                int v_row = m_idx;
                int v_col = output_d_idx;
                float v_val = fourDimRead(V, b, h, v_row, v_col, H, N, d);
                val += qkt_val * v_val;
              }
              fourDimWrite(O, b, h, qkt_row_idx, output_d_idx, H, N, d ,val);
            }
            }
       }
   }




    
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

#define TILE_SIZE 16
torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
  
  // Q, K, V are passed in with Shape: (B, H, N, d)
  //QK^t Intermediate Tensor has Shape (N, N)

  //Make O Tensor with Shape (B, H, N, d) 
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

  //Format O, Q, K, and V tensors into 4D vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);

  //Format QK_t Tensor into a 2D vector.
  std::vector<float> QK_t = formatTensor(QK_tTensor);

  // -------- YOUR CODE HERE  -------- //
  for(int b=0; b < B; b++) {
    for(int h=0; h < H; h++) {

      // incorrect
      // for(int q_row_tile_idx=0; q_row_tile_idx < (N+TILE_SIZE-1)/TILE_SIZE; q_row_tile_idx++) {
      //   // K is not transposed so we traverse k by row.
      //   for(int k_row_tile_idx=0; k_row_tile_idx < (N+TILE_SIZE-1)/TILE_SIZE; k_row_tile_idx++ ) {
      //     for(int d_col_tile_idx=0; d_col_tile_idx < (d+TILE_SIZE-1)/TILE_SIZE; d_col_tile_idx++ ) {
      //       for(int tile_row_idx=0; tile_row_idx < TILE_SIZE; tile_row_idx++) {
      //         // int out_row_idx = q_row_tile_idx * TILE_SIZE + tile_row_idx;
      //         for(int tile_col_idx=0; tile_col_idx < TILE_SIZE; tile_col_idx++) {
      //           int q_row_idx =q_row_tile_idx * TILE_SIZE + tile_row_idx; 
      //           int k_row_idx = k_row_tile_idx * TILE_SIZE + tile_row_idx;
      //           int d_idx = d_col_tile_idx *TILE_SIZE + tile_col_idx;
      //           if(q_row_idx < N  && k_row_idx < N && d_idx < d) {
      //             float q_tile_val = fourDimRead(Q, b, h, q_row_idx, d_idx, H, N, d);
      //             float k_tile_val = fourDimRead(K, b, h, k_row_idx, d_idx, H, N, d);
      //             float orig_val = twoDimRead(QK_t, q_row_idx, k_row_idx, N);
      //             float val = q_tile_val * k_tile_val + orig_val;
      //             twoDimWrite(QK_t, q_row_idx, k_row_idx, N, val );
      //           }
      //         }
      //       }
      //     }

      //   }
      // }
      //

      // correct
      // for(int row_tile_idx=0; row_tile_idx < (N+TILE_SIZE-1)/TILE_SIZE; row_tile_idx++) {
      //   for(int col_tile_idx=0; col_tile_idx < (N+TILE_SIZE-1)/TILE_SIZE; col_tile_idx++) {
      //     for(int k_tile_idx=0; k_tile_idx < (d+TILE_SIZE-1)/TILE_SIZE; k_tile_idx++  ) {
      //       for(int tile_row_idx=0; tile_row_idx < TILE_SIZE; tile_row_idx++) {
      //         for(int tile_col_idx=0; tile_col_idx < TILE_SIZE; tile_col_idx++) {
      //           int row_idx = row_tile_idx * TILE_SIZE + tile_row_idx;
      //           int col_idx = col_tile_idx * TILE_SIZE + tile_col_idx;
      //           if(row_idx >= N || col_idx >= N) {
      //             continue;
      //           }
      //           float sum = twoDimRead(QK_t, row_idx, col_idx, N);

      //           for(int k=0; k < TILE_SIZE; k++) {
      //             int k_idx = k_tile_idx * TILE_SIZE + k;
      //             if(k_idx >= d) break;
      //             float q_val =  fourDimRead(Q,b, h, row_idx, k_idx, H, N, d);
      //             float k_val = fourDimRead(K, b, h, col_idx, k_idx, H, N, d);
      //             sum += q_val * k_val;
      //           }
      //           twoDimWrite(QK_t, row_idx, col_idx, N, sum);
      //         }

      //       }
      //     }
      //   }
      // }

      // correct with local buffer
      for (int row_tile_idx = 0; row_tile_idx < (N + TILE_SIZE - 1) / TILE_SIZE; row_tile_idx++) {
          for (int col_tile_idx = 0; col_tile_idx < (N + TILE_SIZE - 1) / TILE_SIZE; col_tile_idx++) {
              for (int k_tile_idx = 0; k_tile_idx < (d + TILE_SIZE - 1) / TILE_SIZE; k_tile_idx++) {

                  // Buffers for tile data
                  float Q_tile[TILE_SIZE][TILE_SIZE];
                  float K_tile[TILE_SIZE][TILE_SIZE];

                  // Preload Q and K tiles into local buffers
                  for (int tile_row_idx = 0; tile_row_idx < TILE_SIZE; tile_row_idx++) {
                      int row_idx = row_tile_idx * TILE_SIZE + tile_row_idx;
                      if (row_idx >= N) continue; // Skip out-of-bound rows

                      for (int k = 0; k < TILE_SIZE; k++) {
                          int k_idx = k_tile_idx * TILE_SIZE + k;
                          if (k_idx < d) {
                              Q_tile[tile_row_idx][k] = fourDimRead(Q, b, h, row_idx, k_idx, H, N, d);
                          } else {
                              Q_tile[tile_row_idx][k] = 0.0f; // Fill with zero if out-of-bounds
                          }
                      }
                  }

                  for (int tile_col_idx = 0; tile_col_idx < TILE_SIZE; tile_col_idx++) {
                      int col_idx = col_tile_idx * TILE_SIZE + tile_col_idx;
                      if (col_idx >= N) continue; // Skip out-of-bound columns

                      for (int k = 0; k < TILE_SIZE; k++) {
                          int k_idx = k_tile_idx * TILE_SIZE + k;
                          if (k_idx < d) {
                              K_tile[tile_col_idx][k] = fourDimRead(K, b, h, col_idx, k_idx, H, N, d);
                          } else {
                              K_tile[tile_col_idx][k] = 0.0f; // Fill with zero if out-of-bounds
                          }
                      }
                  }

                  // Compute the dot product for the current tile
                  for (int tile_row_idx = 0; tile_row_idx < TILE_SIZE; tile_row_idx++) {
                      int row_idx = row_tile_idx * TILE_SIZE + tile_row_idx;
                      if (row_idx >= N) continue; // Skip out-of-bound rows

                      for (int tile_col_idx = 0; tile_col_idx < TILE_SIZE; tile_col_idx++) {
                          int col_idx = col_tile_idx * TILE_SIZE + tile_col_idx;
                          if (col_idx >= N) continue; // Skip out-of-bound columns

                          float sum = twoDimRead(QK_t, row_idx, col_idx, N);

                          // Unrolled loop for vectorized dot product
                          for (int k = 0; k < TILE_SIZE; k++) {
                              sum += Q_tile[tile_row_idx][k] * K_tile[tile_col_idx][k];
                          }

                          twoDimWrite(QK_t, row_idx, col_idx, N, sum);
                      }
                  }
              }
          }
      }



      // also correct
//       for (int q_row_tile_idx = 0; q_row_tile_idx < (N + TILE_SIZE - 1) / TILE_SIZE; q_row_tile_idx++) {
//     for (int k_row_tile_idx = 0; k_row_tile_idx < (N + TILE_SIZE - 1) / TILE_SIZE; k_row_tile_idx++) {
//         for (int d_col_tile_idx = 0; d_col_tile_idx < (d + TILE_SIZE - 1) / TILE_SIZE; d_col_tile_idx++) {
//             for (int tile_row_idx = 0; tile_row_idx < TILE_SIZE; tile_row_idx++) {
//                 for (int tile_col_idx = 0; tile_col_idx < TILE_SIZE; tile_col_idx++) {
//                     int q_row_idx = q_row_tile_idx * TILE_SIZE + tile_row_idx;
//                     int k_row_idx = k_row_tile_idx * TILE_SIZE + tile_col_idx; // Fix indexing
//                     for (int d_idx = d_col_tile_idx * TILE_SIZE; d_idx < (d_col_tile_idx + 1) * TILE_SIZE; d_idx++) {
//                         if (q_row_idx < N && k_row_idx < N && d_idx < d) {
//                             float q_tile_val = fourDimRead(Q, b, h, q_row_idx, d_idx, H, N, d);
//                             float k_tile_val = fourDimRead(K, b, h, k_row_idx, d_idx, H, N, d);
//                             float orig_val = twoDimRead(QK_t, q_row_idx, k_row_idx, N);
//                             float val = q_tile_val * k_tile_val + orig_val;
//                             twoDimWrite(QK_t, q_row_idx, k_row_idx, N, val);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }




      for(int row_idx=0; row_idx < N; row_idx++) {
        std::vector<double> tmp_row_res(N, 0.0);
        double row_sum = 0.0;
        for(int cold_idx=0; cold_idx < N ;cold_idx++) {
           float val = twoDimRead(QK_t, row_idx, cold_idx, N);
          double exp_val = std::exp(val);
          row_sum += exp_val;
          tmp_row_res[cold_idx] = exp_val;
        }
        for(int cold_idx=0; cold_idx < N ; cold_idx++) {
          float prob = tmp_row_res[cold_idx] / row_sum;
          twoDimWrite(QK_t, row_idx, cold_idx, N, prob);
        }
      }

      for(int qkt_row_tile_idx=0; qkt_row_tile_idx < (N+TILE_SIZE-1)/TILE_SIZE; qkt_row_tile_idx++) {
        for(int output_d_tile_idx=0; output_d_tile_idx < (d+TILE_SIZE-1)/TILE_SIZE; output_d_tile_idx++) {

          for(int k_tile_idx=0; k_tile_idx < (N+TILE_SIZE-1)/TILE_SIZE; k_tile_idx++) {
            for(int tile_row_idx=0; tile_row_idx < TILE_SIZE; tile_row_idx++) {
                int out_row_idx = qkt_row_tile_idx * TILE_SIZE + tile_row_idx;
              if(out_row_idx >= N) continue;
              for(int tile_col_idx=0; tile_col_idx < TILE_SIZE; tile_col_idx++) {
                int out_col_idx = output_d_tile_idx * TILE_SIZE + tile_col_idx;
                if( out_col_idx >= d) continue;

                float sum = fourDimRead(O, b, h, out_row_idx, out_col_idx, H, N, d );
                for(int k=0; k < TILE_SIZE; k++) {
                  int k_idx = k_tile_idx * TILE_SIZE + k;
                  if(k_idx >= N) break;
                  float qkt_val = twoDimRead(QK_t, out_row_idx, k_idx, N);
                  float v_val = fourDimRead(V, b, h, k_idx, out_col_idx, H, N, d);
                  sum += qkt_val * v_val; 
                }
                fourDimWrite(O, b, h, out_row_idx, out_col_idx, H, N, d, sum);
              }
            }
          }
        }
      }


    }

  }


  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
  return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
              int B, int H, int N, int d){

  // Q, K, V are passed in with Shape: (B, H, N, d)

  //Make O Tensor with Shape (B, H, N, d)
  //and O Row Tensor with Shape (N)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
  at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

  //Format Y, Q, K, and V tensors into 4D vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);
  
  //Format ORow Tensor into a 1D vector
  // You can simply access this as ORow[i]
  std::vector<float> ORow = formatTensor(ORowTensor);


  // -------- YOUR CODE HERE  -------- //
  // We give you a template of the first three loops for your convenience
  //loop over batch

  #pragma omp parallel for collapse(3)
  for (int b = 0; b < B; b++){
    //loop over heads
    for (int h = 0; h < H; h++){
        for (int q_row_idx = 0; q_row_idx < N ; q_row_idx++){

  // YRow is moved inside so each OpenMP thread gets a local copy.
            at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
            std::vector<float> ORow = formatTensor(ORowTensor);
  //YOUR CODE HERE
        for(int k_row_idx=0; k_row_idx < N; k_row_idx++) {
          float val = 0.0;
          for(int d_idx=0; d_idx < d ;d_idx++ ) {
            int q_row = q_row_idx;
            int q_col = d_idx;
            int k_row = k_row_idx;
            int k_col = d_idx;
            float q_val = fourDimRead(Q, b, h, q_row, q_col, H, N, d);
            float k_val = fourDimRead(K, b, h, k_row, k_col, H, N, d) ;
            val += q_val * k_val;
          }
          ORow[k_row_idx] = val;
          // std::cout << "krowidx: " << k_row_idx << " val: " << ORow[k_row_idx] << std::endl;

        }
        // softmax
        std::vector<float> tmp_row_res(N, 0.0);
        float sum = 0.0;
        for(int i=0; i < N; i++) {
          ORow[i]  = std::exp(ORow[i]) ;
          sum += ORow[i];
          // tmp_row_res[i] = exp_val;
        }
        for(int i=0; i < N; i++) {
          float prob = ORow[i]  /  sum;
          ORow[i] = prob;
          // std::cout << "softmax col: "  << i << " val: " << ORow[i] << std::endl;
        }

        for(int v_col_idx=0; v_col_idx < d; v_col_idx++) {
          float sum =0.0;
          for(int v_row_idx=0; v_row_idx < N; v_row_idx++) {
            float v_val = fourDimRead(V, b, h, v_row_idx, v_col_idx, H, N ,d);
            sum += v_val * ORow[v_row_idx];
          }
          // std::cout << "vcold_idx" << v_col_idx << "val: " << sum << std::endl;
          fourDimWrite(O, b, h, q_row_idx, v_col_idx, H, N, d, sum);
        }
          
        
      }
    }
  }
    

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
  return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //



torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
      
  // Q, K, V are passed in with Shape: (B, H, N, d)
  // Sij, Pij are passed in with Shape: (Br, Bc)
  // Kj, Vj are passed in with Shape: (Bc, d)
  // Qi, Oi, and PV  are passed in with Shape: (Br, d)
  // L in passed in with Shape: (N)
  // Li, Lij, and Lnew are passed in with shape (Br)

  //Make O Tensor with Shape (B, H, N, d)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
 
  //Format All Tensors into Vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);
  std::vector<float> Sij = formatTensor(SijTensor); //clear
  std::vector<float> Pij = formatTensor(PijTensor); //clear
  std::vector<float> Kj = formatTensor(KjTensor); // clear
  std::vector<float> Vj = formatTensor(VjTensor); // clear
  std::vector<float> Qi = formatTensor(QiTensor); // clear
  std::vector<float> Oi = formatTensor(OiTensor); //clear
  std::vector<float> l = formatTensor(LTensor); // This should be cleared
  std::vector<float> PV = formatTensor(PVTensor);
  std::vector<float> li = formatTensor(LiTensor);
  std::vector<float> lij = formatTensor(LijTensor);
  std::vector<float> lnew = formatTensor(LnewTensor);

  // std::cout << "br:" << Br << " bc:" << Bc <<std::endl;
  // -------- YOUR CODE HERE  -------- //
  for(int b=0; b < B; b++ ) {
    for(int h=0; h < H; h++) {

    std::fill(l.begin(), l.end(), 0.0f);
    std::fill(lnew.begin(), lnew.end(), 0.0f);
    std::fill(lij.begin(), lij.end(), 0.0f);
  for(int k_block_idx=0; k_block_idx < (N+Bc-1)/Bc; k_block_idx++) {
    std::fill(Kj.begin(), Kj.end(), 0.0f);
    std::fill(Vj.begin(), Vj.end(), 0.0f);
    // load Kj, Vj into local memory blocks.
    for(int j=0; j < Bc; j++) {
      int j_row = k_block_idx * Bc + j;
      if(j_row >= N) continue;
      for(int d_idx =0; d_idx < d; d_idx++) {
        float k_val = fourDimRead(K, b, h, j_row, d_idx, H, N, d);
        float v_val = fourDimRead(V, b, h, j_row, d_idx, H, N, d);
        twoDimWrite(Kj, j, d_idx, d, k_val);
        twoDimWrite(Vj, j, d_idx, d, v_val);
          // std::cout<< "j:" << j_row << " col:" << d_idx << "kj:" << k_val << " vj:" << v_val << std::endl;
      }
    }

    for(int q_block_idx=0; q_block_idx < (N+Br-1)/Br; q_block_idx++) {
      std::fill(Qi.begin(), Qi.end(), 0.0f);
      std::fill(Oi.begin(), Oi.end(), 0.0f);
      std::fill(Sij.begin(), Sij.end(), 0.0f);
      std::fill(Pij.begin(), Pij.end(), 0.0f);


      // load Qi, Oi, li into local memory blocks
      for(int br_idx=0; br_idx < Br; br_idx++ ) {
        int q_row_idx = q_block_idx * Br + br_idx; 
        if(q_row_idx >= N ) continue;
        for(int d_idx=0; d_idx < d; d_idx++) {
          float q_val = fourDimRead(Q, b, h, q_row_idx, d_idx, H, N, d);
          float o_val = fourDimRead(O, b, h, q_row_idx , d_idx, H, N, d);
          twoDimWrite(Qi, br_idx, d_idx, d, q_val);
          twoDimWrite(Oi, br_idx, d_idx, d, o_val);
            // std::cout << "q_row_idx:" << q_row_idx << " d_idx:" << d_idx << " Qi:" << q_val << " Oi:" << o_val <<std::endl;

        }
        float l_val = l[q_row_idx];
        li[br_idx] = l_val;
            // std::cout << "li:" << l_val << std::endl;

      }

      // compute Sij  = Qi * Kj_T (Br x Bc) 
      for(int br_idx=0; br_idx < Br; br_idx++) {
        for(int bc_idx=0; bc_idx < Bc; bc_idx++) {
          float sum = 0.0;
          for(int d_idx=0; d_idx < d; d_idx++) {
            float q_val = twoDimRead(Qi, br_idx, d_idx, d);
            float k_val = twoDimRead(Kj, bc_idx, d_idx, d);
            sum += q_val * k_val;

          }
          twoDimWrite(Sij, br_idx, bc_idx, Bc, sum);
              // std::cout << "sij, br:" << br_idx << " bc:" << bc_idx << " val:" << sum << std::endl;
        }
      }

      // Compute Pij = exp(Sij) of size (Br x Bc)
      for(int br_idx=0; br_idx < Br; br_idx++) {
        for(int bc_idx=0; bc_idx < Bc; bc_idx++) {
          float exp_val = std::exp(twoDimRead(Sij, br_idx, bc_idx, Bc));
          twoDimWrite(Pij, br_idx, bc_idx, Bc, exp_val);
        }
      }

      // Compute lij = rowsum(Pij) of size (Br)
      for(int br_idx=0; br_idx < Br; br_idx++) {
        float sum = 0.0;
        for(int bc_idx=0; bc_idx < Bc; bc_idx++) {
          sum += twoDimRead(Pij, br_idx, bc_idx, Bc);
        }
        lij[br_idx] = sum;
        // compute lnew = li + lij
        lnew[br_idx] = li[br_idx] + lij[br_idx];

      }



      // Compute Oi = (liOi + PijVj)/ lnew
      for(int br_idx=0; br_idx < Br; br_idx++) {
        for(int d_idx=0; d_idx < d; d_idx++) {
          float pv_sum =0.0;
          for(int bc_idx=0; bc_idx < Bc; bc_idx++) {
            int p_row = br_idx;
            int p_col = bc_idx;
            int v_row = bc_idx;
            int v_col = d_idx;
            pv_sum += twoDimRead(Pij, p_row, p_col, Bc) * twoDimRead(Vj, v_row, v_col, d);

          }
          // twoDimWrite(PV, br_idx, d_idx, d, pv_sum);

          float li_Oi_val = li[br_idx] * twoDimRead(Oi, br_idx, d_idx, d);
          float new_sum = pv_sum + li_Oi_val;
          float new_Oi_val = new_sum / lnew[br_idx];
          twoDimWrite(Oi, br_idx, d_idx, d, new_Oi_val);
        }
      }

      // Write Oi and lnew back to O and l in main memory
      for(int br_idx=0; br_idx < Br; br_idx++) {
        int O_row = q_block_idx * Br + br_idx;
        if(O_row >= N) continue;
        for(int d_idx=0; d_idx < d; d_idx++) {
          float Oi_val = twoDimRead(Oi, br_idx, d_idx, d);
                  int O_col = d_idx;
          fourDimWrite(O, b, h, O_row, O_col, H, N, d, Oi_val);

        }

        l[O_row] = lnew[br_idx];
          // l[O_row] += lij[br_idx];

      }


    }
  }

    }

  }


  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
  return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}

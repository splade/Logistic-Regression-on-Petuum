#include <petuum_ps/include/petuum_ps.hpp>
#include <atomic>
#include <stdint.h>
#include <vector> // TODO: which is better, <valarray>?
#include <valarray>
#include <boost/thread/tss.hpp>

namespace lrl2 {

inline float dot_product(std::valarray<float> &record, std::valarray<float> &weights);
inline float inv_sigmoid(std::valarray<float> &record, std::valarray<float> &weights, float label);
inline float sigmoid(std::valarray<float> &record, std::valarray<float> &weights, float label); 

struct LRL2SolverConfig {
  float stepsize; // step TODO: dynamic step?
  int eval_interval; // # iterations between each primal eval
  int32_t aug_dim; // (augmented) data dimension. #weigths + 1
  float lambda;

  //TODO: study how to access data partition.
  int32_t data_global_idx_begin;

  int32_t max_iter;
  int32_t batch_size;
  int32_t client_rank;

  int32_t num_threads;

  std::string train_data_file;
  std::string test_data_file;
};

class LRL2Solver {
  public:
    LRL2Solver(petuum::TableGroup<float> *table_group,
	       petuum::DenseTable<float> *w_table,
	       petuum::DenseTable<float> *w_snapshot_table,
	       petuum::DenseTable<float> *loss_table,
	       const LRL2SolverConfig& config);

    void Init();

    void Solve();

  private:
    //TODO: Why dai wei's code uses pointer instead of ref?
    void ReadData(const std::string& data_file,
		  std::vector<std::valarray<float> >& X,
		  std::vector<int32_t>& Y);

    void SolveOne(int32_t data_idx);
    int32_t SolveBatch(int32_t local_idx);
    void SolveBatchPartialWeights(int32_t data_idx);

    float EvaluateObjLoc();

    void EvaluateObjDist(int evaluation_count);

    float ComputeTestError(std::vector<std::valarray<float> >& X_test,
			   std::vector<int32_t>& Y_test, std::valarray<float>& w); 


    petuum::TableGroup<float>* _table_group;
    petuum::DenseTable<float>* _w_table;
    petuum::DenseTable<float>* _w_snapshot_table;
    petuum::DenseTable<float>* _loss_table;

    LRL2SolverConfig _config;
    int32_t _num_data;

    int32_t _aug_dim; //data dim + 1
    float _lambda; //regularization parameter
   
    //TODO:now is const stepsise, hope to change to dynamic stepsize 
    float _stepsize;

    std::vector<std::valarray<float> > _X;
    std::vector<int> _Y;
    //std::vector<std::vector<float> > _X;
    //std::vector<float> _Y;


    std::atomic<int32_t> _thread_counter;

    struct LRL2SolverThreadData {
      int32_t thread_id;
      //[begin, end)
      int32_t data_idx_begin;
      int32_t data_idx_end;

      //TODO:this version should update all weights, 
      //next version will only update weights [begin, end)
      int32_t w_idx_begin;
      int32_t w_idx_end;
    };

    boost::thread_specific_ptr<LRL2SolverThreadData> _solver_thread_data;

};
}; //namespace lrl2

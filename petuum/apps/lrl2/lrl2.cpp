#include "lrl2.hpp"
#include <algorithm>
#include <glog/logging.h>
#include <stdio.h>
#include <string>
#include <cmath>
//#include <boost/unordered_set.hpp>
#include <iostream>
#include <time.h>
#include <stdlib.h>

namespace lrl2 {

LRL2Solver::LRL2Solver(petuum::TableGroup<float>* table_group,
		       petuum::DenseTable<float>* w_table,
		       petuum::DenseTable<float>* w_snapshot_table,
		       petuum::DenseTable<float>* loss_table,
		       const LRL2SolverConfig& config) :
  _config(config), _thread_counter(0) {
  CHECK_NOTNULL(table_group);
  CHECK_NOTNULL(w_table);
  CHECK_NOTNULL(w_snapshot_table);
  CHECK_NOTNULL(loss_table);
  _table_group = table_group;
  _w_table = w_table;
  _w_snapshot_table = w_snapshot_table;
  _loss_table = loss_table;

  CHECK_GT(config.lambda, 0);
  CHECK_GT(config.aug_dim, 1);
  _aug_dim = config.aug_dim;
  _lambda = config.lambda;
  _stepsize = config.stepsize;

  //data_global_idx_begin_ = config.data_global_idx_begin;
  //data_global_idx_end_ = config.data_global_idx_end;
}

void LRL2Solver::Init() {
  LOG(INFO) << "reading dataset from " <<_config.train_data_file;
  ReadData(_config.train_data_file, _X, _Y);
  LOG(INFO) << "done reading dataset from " <<_config.train_data_file;
  srand(time(NULL));
}

void LRL2Solver::Solve() {
  // Register thread on PS.
  CHECK_EQ(0, _table_group->RegisterThread()) << "Failed to register thread";
  // Needed, this thread interacts with Tables
  CHECK_EQ(0, _table_group->RegisterExecutionThread())
    << "Failed to register execution thread.";

  //LOG(INFO) << "begin Solve";
  int32_t num_threads = _config.num_threads;
  int32_t num_data_per_thread = static_cast<int32_t>((_num_data + num_threads - 1)/ num_threads);
  //int32_t num_data_per_thread = static_cast<int32_t>(_num_data / num_threads);

  // Initialize thread specific data.
  if (!_solver_thread_data.get()) {
    _solver_thread_data.reset(new LRL2SolverThreadData());
    _solver_thread_data->thread_id = _thread_counter++;

    // Get data range for this thread.
    _solver_thread_data->data_idx_begin = _solver_thread_data->thread_id * num_data_per_thread;
    if (_solver_thread_data->thread_id == _config.num_threads - 1) {
      //TODO:what's this?
      _solver_thread_data->data_idx_end = _config.data_global_idx_begin + _num_data;
    }
    else {
      _solver_thread_data->data_idx_end = _solver_thread_data->data_idx_begin + num_data_per_thread;
    }
  }

  int idx_begin = _solver_thread_data->data_idx_begin;
  int idx_end = _solver_thread_data->data_idx_end;
  LOG(INFO) << "Thread " << _solver_thread_data->thread_id << " is solving "
    << idx_begin << " ~ " << idx_end;
  int max_iter = _config.max_iter;
  
  int evaluation_count = 0;
  bool converged = false;

  //training
  for(int iter = 0; iter < max_iter && !converged; iter++){
    //TODO:replace with batch, partial weights in the future
    int data_idx = rand() % (idx_end - idx_begin) + idx_begin;
    SolveOne(data_idx);
    _table_group->Iterate();


    // Compute objective
    if (iter % _config.eval_interval == 0) {
      if (_config.client_rank == 0 &&
          _solver_thread_data->thread_id == 0) {
        // Global head.
        LOG(INFO) << "Taking snapshot for outer iter = " << iter;
      }
      EvaluateObjDist(evaluation_count);
      if (_solver_thread_data->data_idx_begin == 0) {
        // Head thread. Take an approximate snapshot of w.
        petuum::DenseRow<float>& w_row = _w_table->GetRowUnsafe(0);
        for (int dim = 0; dim < _aug_dim; ++dim) {
          _w_snapshot_table->Put(evaluation_count, dim, w_row[dim]);
        }
      }
      ++evaluation_count;
    }

  }

  _table_group->GlobalBarrier();

  //output inter weights
  if (_config.client_rank == 0 && _solver_thread_data->thread_id == 0) {
    // Global head thread. Compute distributed primal objective.
    LOG(INFO) << "iter-# objectives";
    for (int eval_counter = 0; eval_counter < evaluation_count;
        ++eval_counter) {
      float loss = _loss_table->Get(eval_counter, 0);
      petuum::DenseRow<float>& w_snapshot_row =
        _w_snapshot_table->GetRowUnsafe(eval_counter);
      std::valarray<float> w(_aug_dim);
      for (int i = 0; i < _aug_dim; ++i) {
        w[i] = w_snapshot_row[i];
      }
      float obj = 0.5 * dot_product(w, w) + loss;
      // Use std out to avoid prefixes from Google logging.
      std::cout << eval_counter * _config.eval_interval << " "
        << obj << std::endl;
    }
  }
  //test

  _table_group->DeregisterThread();
  //LOG(INFO) << "end Solve";
}

//modify dai wei's code, change vector to valarray
void LRL2Solver::ReadData(const std::string& data_file,
			  std::vector<std::valarray<float> >& X,
			  std::vector<int32_t>& Y) {
  X.clear();
  Y.clear();
  char *line = NULL, *ptr = NULL;
  size_t num_bytes;
  FILE *data_stream = fopen(data_file.c_str(), "r");
  if(data_stream==NULL)
    LOG(INFO) << "error "<<errno; 
  CHECK_NOTNULL(data_stream);
  LOG(INFO) << "Reading from data file " << data_file;
  int num_data = 0;
  //int aug_dim_ = DATA_DIM + 1;
  //int max_feature_id = 0;
  bool convert_label_notice = false;
  while (getline(&line, &num_bytes, data_stream) != -1) {
    // stat of a word
    int32_t label, feature_id;
    float feature_val;
    ptr = line; // point to the start
    sscanf(ptr, "%d", &label);
    //Check binary label is {0, 1} or {-1, 1}
    CHECK_LE(label, 1) << "data " << _num_data << " has label " << label
      << " out of range.";
    CHECK_LE(-1, label) << "data " << _num_data << " has label " << label
      << " out of range.";
    //Convert {0, 1} label to {-1, 1} labels.
    if (label == 0) {
      if (!convert_label_notice) {
        LOG(INFO) << "converting label 0 to -1";
      }
      label = -1;
    }
    Y.push_back(label);
    std::valarray<float> datum(_aug_dim);
    // Last dimension is always 1
    datum[0] = 1.f;
    while (*ptr != '\n') {
      while (*ptr != ' ') ++ptr; // goto next space
      // read a feature_id:feature_val pair
      sscanf(++ptr, "%d:%f", &feature_id, &feature_val);
      //max_feature_id = std::max(feature_id, max_feature_id);
      datum[feature_id] = feature_val;
      while (*ptr != ' ' && *ptr != '\n') ++ptr; // goto next space or \n
    }
    X.push_back(datum);
    ++num_data;
    LOG_IF(INFO, num_data % 100000 == 0) << "Read data " << _num_data;
  }
  //LOG(INFO) << "max_feature_id = " << max_feature_id;
  free(line);
  CHECK_EQ(0, fclose(data_stream)) << "Failed to close file " << data_file;

  LOG(INFO) << "Read " << _num_data << " data from " << data_file;
}

inline float dot_product(std::valarray<float> &record, std::valarray<float> &weights){
    return (record * weights).sum();
};

inline float inv_sigmoid(std::valarray<float> &record, std::valarray<float> &weights, float label) {
  return 1.f + exp(- dot_product(record, weights) * label);
};

inline float sigmoid(std::valarray<float> &record, std::valarray<float> &weights, float label) {
  return 1.f / (1.f + exp(- dot_product(record, weights) * label));
};

void LRL2Solver::SolveOne(int32_t data_idx) {
  //LOG(INFO)<<"begin solveone";
  petuum::DenseRow<float>& w_row = _w_table->GetRowUnsafe(0);
  std::valarray<float> w(_aug_dim);
  for (int i = 0; i < _aug_dim; i++){
    w[i] = w_row[i];
  }

  std::valarray<float> grad(w.size());
  float sig = sigmoid(_X[data_idx], w, _Y[data_idx]);
  grad += (sig - 1.f) * (float)(_Y[data_idx]) * _X[data_idx];
  grad += _lambda * w;
  grad *= _stepsize;

  //Update weights
  for (int i = 0; i < _aug_dim; i++){
    _w_table->Inc(0, i, -grad[i]);
  }
  //LOG(INFO)<<"end solveone";
}

void LRL2Solver::SolveBatch(int32_t data_idx) {

}

void LRL2Solver::SolveBatchPartialWeights(int32_t data_idx) {
  //TODO:scatter and gather, only handle one part of weights
}

void LRL2Solver::EvaluateObjDist(int evaluation_count) {
  petuum::DenseRow<float>& w_row = _w_table->GetRowUnsafe(0);
  std::valarray<float> w(_aug_dim);
  for (int i = 0; i < _aug_dim; i++){
    w[i] = w_row[i];
  }

  //TODO:What is this?
  int idx_begin = _solver_thread_data->data_idx_begin;
  int idx_end = _solver_thread_data->data_idx_end;
  float loss = 0.f;

  for (int i = idx_begin; i < idx_end; i++) {
    loss += log(inv_sigmoid(_X[i], w, _Y[i]));
  }

  _loss_table->Inc(evaluation_count, 0, loss);
}

float LRL2Solver::EvaluateObjLoc() {
  petuum::DenseRow<float>& w_row = _w_table->GetRowUnsafe(0);
  std::valarray<float> w(_aug_dim);
  for (int i = 0; i < _aug_dim; i++){
    w[i] = w_row[i];
  }

  //TODO:What is this?
  int idx_begin = _solver_thread_data->data_idx_begin;
  int idx_end = _solver_thread_data->data_idx_end;
  float loss = 0.f;

  for (int i = idx_begin; i < idx_end; i++) {
    loss += log(inv_sigmoid(_X[i], w, _Y[i]));
  }
  loss += _lambda * 0.5f * dot_product(w, w);

  return loss;
}

float LRL2Solver::ComputeTestError(std::vector<std::valarray<float> >& X_test,
				   std::vector<int32_t>& Y_test, std::valarray<float>& w) {
  int num_tests = X_test.size();
  int error_count = 0;
  for (int i = 0; i < num_tests; i++){
    int prediction = (dot_product(w, X_test[i]) > 0) ? 1 : -1;
    error_count += (prediction == Y_test[i]) ? 0 : 1; 
  }
  return static_cast<float>(error_count) / static_cast<float>(num_tests);
}

} //namespace lrl2

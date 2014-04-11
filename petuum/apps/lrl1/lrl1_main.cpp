#include "lrl1.hpp"
#include <petuum_ps/include/petuum_ps.hpp>
#include <petuum_ps/util/utils.hpp>
#include <zmq.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <vector>
#include <valarray>

// Petuum Inputs:
DEFINE_string(server_file, "",
    "Path to file containing server ip:port.");
DEFINE_string(config_file, "",
    "Path to .cfg file containing client configs.");
DEFINE_int32(num_threads, 1, "Number of worker threads");
DEFINE_int32(client_id, 10000, "ID of a client process");

// LRL1 Inputs:
DEFINE_int32(client_rank, -1, "Rank of a client process");
DEFINE_string(train_data_file, "",
    "File containing document in LibSVM format. Each document is a line.");
DEFINE_string(test_data_file, "",
    "File containing document in LibSVM format. Each document is a line.");
DEFINE_int32(data_global_idx_begin, -1,
    "Start of this data partition using global index.");
//DEFINE_int32(num_data, -1, "Number of data in this partition.");
DEFINE_int32(aug_dim, -1, "Augmented dimension of feature.");
DEFINE_int32(max_iter, 10, "Maximum number of outer iterations.");
DEFINE_int32(batch_size, 10,
    "Number of coordinate descents in each outer iteration");
DEFINE_int32(eval_interval, 1, "# iterations between snapshotting.");
DEFINE_double(lambda, -1, "regularization parameter.");
DEFINE_double(stepsize, 0.001, "updating step size.");
//DEFINE_double(primal_converge_epsilon, 1e-4, "primal convergence criterion.");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "LRL1 started on client " << FLAGS_client_id
    << " server_file: " << FLAGS_server_file
    << " num threads: " << FLAGS_num_threads;

  //CHECK_GT(FLAGS_num_data, 0);
  CHECK_GT(FLAGS_aug_dim, 1);
  CHECK_GE(FLAGS_lambda, 0);
  CHECK_GE(FLAGS_client_rank, 0);
  CHECK_GE(FLAGS_data_global_idx_begin, 0);

  // Initialize TableGroup.
  zmq::context_t *zmq_ctx = new zmq::context_t(1);
  std::vector<petuum::ServerInfo> server_info =
    petuum::GetServerInfos(FLAGS_server_file);
  petuum::TableGroupConfig table_group_config(FLAGS_client_id,
      FLAGS_num_threads, server_info, zmq_ctx);
  petuum::TableGroup<float>& table_group =
    petuum::TableGroup<float>::CreateTableGroup(
        table_group_config);

  // Register the main thread.
  CHECK_EQ(0, table_group.RegisterThread()) << "Failed to register thread.";

  // Read table configs.
  std::map<int, petuum::TableConfig> table_configs_map =
    petuum::ReadTableConfigs(FLAGS_config_file);

  // Construct w table which has only 1 row of dim columns.
  int32_t table_id = 1;
  LOG(INFO) << "Creating table " << table_id;
  LOG_IF(FATAL, table_configs_map.count(table_id) == 0)
    << "Table (table_id = " << table_id << ") is not in the config file "
    << FLAGS_config_file;
  table_configs_map[table_id].num_columns = FLAGS_aug_dim;  // override
  petuum::DenseTable<float>& w_table =
    table_group.CreateDenseTable(table_id,
        table_configs_map[table_id]);

  // Construct w_snapshot table which has # snapshot rows and aug_dim columns.
  table_id = 2;
  LOG(INFO) << "Creating table " << table_id;
  LOG_IF(FATAL, table_configs_map.count(table_id) == 0)
    << "Table (table_id = " << table_id << ") is not in the config file "
    << FLAGS_config_file;
  table_configs_map[table_id].num_columns = FLAGS_aug_dim;  // override
  petuum::DenseTable<float>& w_snapshot_table =
    table_group.CreateDenseTable(table_id,
        table_configs_map[table_id]);

  // Construct loss table which has # snapshot rows and 1 columns.
  table_id = 3;
  LOG(INFO) << "Creating table " << table_id;
  LOG_IF(FATAL, table_configs_map.count(table_id) == 0)
    << "Table (table_id = " << table_id << ") is not in the config file "
    << FLAGS_config_file;
  petuum::DenseTable<float>& loss_table =
    table_group.CreateDenseTable(table_id,
        table_configs_map[table_id]);

  lrl1::LRL1SolverConfig solver_config;
  solver_config.data_global_idx_begin = FLAGS_data_global_idx_begin;
  solver_config.aug_dim = FLAGS_aug_dim;
  solver_config.lambda = (float)FLAGS_lambda;
  solver_config.stepsize= (float)FLAGS_stepsize;
  solver_config.client_rank = FLAGS_client_rank;
  solver_config.eval_interval = FLAGS_eval_interval;
  solver_config.max_iter= FLAGS_max_iter;
  solver_config.batch_size = FLAGS_batch_size;
  solver_config.num_threads = FLAGS_num_threads;
  solver_config.train_data_file = FLAGS_train_data_file;
  solver_config.test_data_file = FLAGS_test_data_file;
  lrl1::LRL1Solver lrl1_solver(&table_group, &w_table, &w_snapshot_table,
      &loss_table, solver_config);
  lrl1_solver.Init();
  LOG(INFO) << "Done reading dataset.";

  // Create FLAGS_num_threads to run lda_sampler.RunSampler()
  boost::thread_group worker_threads;
  for (int i = 0; i < FLAGS_num_threads; ++i) {
    worker_threads.create_thread(
        boost::bind(&lrl1::LRL1Solver::Solve, boost::ref(lrl1_solver)));
  }
  worker_threads.join_all();

  LOG(INFO) << "Done! Shutting down table_group.";

  // Finish and exit
  table_group.DeregisterThread();
  table_group.ShutDown();
  delete zmq_ctx;

  return 0;
}

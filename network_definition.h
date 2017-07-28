//size definition
#define EPOCH 40
#define MINI_BATCH_SIZE 100
#define LEARNING_RATE 0.8
#define REPORT_F "./result/dump"

#define THREAD_NUM 64

#define MODE_RECORD "./record/mode_record"

#define SELECT_MODE 1

#define NUM_LAYER 3
#define HIDDEN_SIZE 120

#define MODE_NUM 3
#define THREAD_MODE_NUM 5

#define TOTAL_NEURONS(net_p)     AC_NEURONS(net_p, net_p->num_layer-1)
#define TOTAL_WEIGHTS(net_p)     AC_WEIGHTS(net_p, net_p->num_layer-2)

#define MAX_CPU 256

#define AC_NEURONS(net_p, L)       (0 > L ? 0 : net_p->ac_neuron[L])
#define AC_WEIGHTS(net_p, L)       (0 > L ? 0 : net_p->ac_weight[L])

#define BIAS(net_p, i, j)          (net_p->bias[AC_NEURONS(net_p, i-1) + j])
//ith layer, jth node to k node weight
#define WEIGHT(net_p, i, j, k)     (net_p->weight[AC_WEIGHTS(net_p, i-1) \
									+ j*net_p->layer_size[i+1] + k])


// ith layer, jth mini_batch, kth node
#define NEURON(net_p, i, j, k)      (net_p->neuron[AC_NEURONS(net_p, i-1)*net_p->mini_batch_size \
									+ net_p->layer_size[i]*(j) + (k)])
#define ZS(net_p, i, j, k)      	(net_p->zs[AC_NEURONS(net_p, i-1)*net_p->mini_batch_size \
									+ net_p->layer_size[i]*(j) + (k)])
#define ERROR(net_p, i, j, k)      	(net_p->error[AC_NEURONS(net_p, i-1)*net_p->mini_batch_size \
									+ net_p->layer_size[i]*(j) + (k)])

#define DATA_TRAIN_Q(net, i, j)		(net->train_q[net->layer_size[0]*i + j])
#define DATA_TRAIN_A(net, i)		(net->train_a[i])
#define DATA_TEST_Q(net, i, j)		(net->test_q[net->layer_size[0]*i + j])
#define DATA_TEST_A(net, i)			(net->test_a[i])

#define WHILE						while(1)

#ifdef HBWMODE
#define malloc(x)       hbw_malloc(x)
#define calloc(v, x)    hbw_calloc(v, x)
#define free(x)         hbw_free(x)
#else
#define malloc(x)       malloc(x)
#define calloc(v, x)    calloc(v, x)
#define free(x)         free(x)
#endif

enum DATA_T {BIAS, WEIGHT, ERROR, ZS, NEURON};
#define NUMBER_OF_NETWORK 2
#define NUMBER_OF_ACTION 2
#define NUMBER_OF_LOSS 1
#define NUMBER_OF_OPTIMIZER 1
#define NUMBER_OF_DATA 1


const char * network_Name[NUMBER_OF_NETWORK] = {"fully connected","CNN"};
const char * action_Name[NUMBER_OF_ACTION] = {"sigmoid","relu"};
const char * loss_Name[NUMBER_OF_LOSS] = {"mean_squared_error"};
const char * optimizer_Name[NUMBER_OF_OPTIMIZER] = {"sgd"};
const char * data_Name[NUMBER_OF_DATA] = {"mnist"};

struct network {

    int network_info; //this is information number that  what kind of network
    int action_info;
    int loss_info;
    int optimizer_info;
    int data_info;

    int nr_thread;
	int num_layer;
	int *layer_size;

	char *train_q_name, *train_a_name;
	char *test_q_name , *test_a_name;

	char *report_file;

	unsigned int nr_train_data;
	unsigned int nr_test_data;

	double *train_q, *test_q;
	int *train_a, *test_a;

	double *neuron;
	double *zs;
	double *error;
	double *bias;
	double *weight;

	int *ac_weight;
	int *ac_neuron;

	int *mode;
	int *thread;

	int *record_random;

	double learning_rate;
	int mini_batch_size;
	int epoch;

	int best_recog;

	double cost_rate;

	struct timeutils t_feedforward;
	struct timeutils t_back_pass;
	struct timeutils t_backpropagation;

};

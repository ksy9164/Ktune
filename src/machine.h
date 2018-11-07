#include "immintrin.h"
/*** CAUTION *** : net->mode 를 MKL 즉 mode를 0으로 할 때는 network_definition.h 에서 VECTORIZED_DATA 를 undefine 할 것 ***/

int predict(struct network *net);
float sigmoid_prime(float z);
float sigmoid(float z);
void train(struct network *net);
void init(struct network *net);
float randn(void);
void feedforward(struct network *net);
void feedforward00(struct network *net);
void feedforward01(struct network *net);
void feedforward02(struct network *net);
void feedforward03(struct network *net);
void feedbackward(struct network *net);
void feedbackward00(struct network *net);
void feedbackward01(struct network *net);
void feedbackward02(struct network *net);
void feedbackward03(struct network *net);
void update_parameters(struct network *net);
void update_parameters00(struct network *net);
void update_parameters01(struct network *net);
void update_parameters02(struct network *net);
void update_parameters03(struct network *net);
void cost_report(struct network * net );
void report(struct network *net);

//int layersize[NUM_LAYER] = {INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE};

#define VECTOR_LEN 16

__m512 v1p;
__m512 v1n;

__m512 Vsigmoid(__m512 z)
{
    __m512 vz = _mm512_mul_ps (v1n, z);
    vz = _mm512_exp_ps (vz);
    vz = _mm512_add_ps (v1p, vz);
    return  _mm512_rcp14_ps (vz);
}

__m512 Vsigmoid_prime(__m512 z)
{
    __m512 vz = _mm512_mul_ps (v1n, z);
    __m512 vd = _mm512_exp_ps (vz);
    vz = _mm512_add_ps (v1p, vz);
    vz = _mm512_mul_ps (vz, vz);
    vz = _mm512_div_ps (vd, vz);
    return  vz;
}

float sigmoid(float z)
{
    return (1/(1 + exp(-z)));
}

float sigmoid_prime(float z)
{
    return sigmoid(z)*(1-sigmoid(z));
}


float randn(void)
{
    float v1, v2, s;

    do {
        v1 =  2 * ((float) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지
        v2 =  2 * ((float) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
        s = v1 * v1 + v2 * v2;
    } while (s >= 1 || s == 0);

    s = sqrt( (-2 * log(s)) / s );

    return v1 * s;
}


void init(struct network *net)
{
    int i,j,k;
    int before_ac_weights = 0;
    int before_ac_neurals = 0;

    timeutils *t_feedforward = &net->t_feedforward;
    timeutils *t_feedbackward = &net->t_feedbackward;
    timeutils *t_update_parameters = &net->t_update_parameters;

    v1p = _mm512_set1_ps(1.0);
    v1n = _mm512_set1_ps(-1.0);

    net->best_recog = 0.0;
    TIMER_INIT(t_feedforward); //시간 초기화
    TIMER_INIT(t_feedbackward);
    TIMER_INIT(t_update_parameters);
    
    net->cost_rate = 0;

    net->ac_weight = (int *) malloc(sizeof(float) * net->num_layer);
    net->ac_neuron = (int *) malloc(sizeof(float) * net->num_layer);

    net->thread = (int *)malloc(sizeof(int) * THREAD_MODE_NUM);
    net->mode = (int *)malloc(sizeof(int)*MODE_NUM);
    net->thread_arr = (float *)malloc(sizeof(float)*(256+1));
    net->record_random = (int *)malloc(sizeof(int) *net->mini_batch_size);

//  net->train_q_name = TRAIN_Q;
//  net->train_a_name = TRAIN_A;
//  net->test_q_name = TEST_Q;
//  net->test_a_name = TEST_A;
    net->report_file = REPORT_F;

    //init mode & thread
    for(i=0;i<THREAD_MODE_NUM;i++)
        net->thread[i] =THREAD_NUM;
/*  
    for(i=0;i<MODE_NUM;i++)
    {   net->mode[i] =0;
        if(i==1)net->mode[i] = 1;
    }
*/

//    net->mode[0] =3;
//    net->mode[1] =2;
//    net->mode[2] =2;

    net->mode[0] =0;
    net->mode[1] =0;
    net->mode[2] =0;
    for (i = 0; i < net->num_layer; i++) {
        net->ac_neuron[i] = net->layer_size[i] + before_ac_neurals;//ac_neuron은 여태 누적한 neuron갯수..
        before_ac_neurals = net->ac_neuron[i];

        if (i == net->num_layer-1)
            continue;

        net->ac_weight[i] = net->layer_size[i] * net->layer_size[i+1] + before_ac_weights; //ac_weight는 여태 누적한 weight 의 갯수..
        before_ac_weights = net->ac_weight[i]; 
    }
    if(net->mode[0]>1)
    {
        net->neuron = (float *) _mm_malloc(sizeof(float) * net->mini_batch_size * TOTAL_NEURONS(net), 64); //neuron 배열의 크기는 minibatch_size * 총 뉴련의 숫자
        net->zs = (float *) _mm_malloc(sizeof(float) * net->mini_batch_size * TOTAL_NEURONS(net), 64);
        net->error =  (float *) _mm_malloc(sizeof(float) * net->mini_batch_size * TOTAL_NEURONS(net), 64);
        net->bias = (float *) _mm_malloc(sizeof(float) * TOTAL_NEURONS(net), 64);
        net->weight = (float *) _mm_malloc(sizeof(float) * TOTAL_WEIGHTS(net), 64);
    }
    else
    {
        net->neuron = (float *) malloc(sizeof(float) * net->mini_batch_size * TOTAL_NEURONS(net)); //neuron 배열의 크기는 minibatch_size * 총 뉴련의 숫자
        net->zs = (float *) malloc(sizeof(float) * net->mini_batch_size * TOTAL_NEURONS(net));
        net->error =  (float *) malloc(sizeof(float) * net->mini_batch_size * TOTAL_NEURONS(net));
        net->bias = (float *) malloc(sizeof(float) * TOTAL_NEURONS(net));
        net->weight = (float *) malloc(sizeof(float) * TOTAL_WEIGHTS(net));
    }

    for (i = 0; i < TOTAL_WEIGHTS(net); i++) {
        net->weight[i] = randn();
    }

    for (i = 0; i < TOTAL_NEURONS(net); i++) {
        net->bias[i] = randn();
    }
}

void load_kth_data(struct network *net, int s_index, int k){
    int first_layer_size = AC_NEURONS(net, 0);//input  size
    int last_layer_size = net->layer_size[net->num_layer-1];        //output size
    for (int l = 0; l < first_layer_size; l++)                    //l은 28*28 까지 증가합니다
        NEURON(net, 0, k, l) = DATA_TRAIN_Q(net, s_index, l); //s_index 번째 데이터를 가져옵니다 그것을 net->neuron[net->layer_size[0]*(k) + (l)] 에 넣습니다.
                                                                        //즉 neuron 배열에 차곡차곡 랜덤한 인풋값을 넣습니다.
    for (int l = 0; l < last_layer_size; l++)
        ERROR(net, net->num_layer-1, k, l) = 0.0;
        // copy output to error array
    ERROR(net, net->num_layer-1, k, DATA_TRAIN_A(net, s_index)) = 1.0; //답안 배열에 1의값 넣습니다.
    net->record_random[k] = s_index;
}



void train(struct network *net)
{

    int i, j, k, l;
    int nr_train = net->nr_train_data;
    int nr_loop = (int)(net->nr_train_data/net->mini_batch_size);   //전체데이터를 미니배치 사이즈 만큼 나눈 수 입니다.(업데이트 할 숫자)
    int first_layer_size = AC_NEURONS(net, 0);//input  size
    int last_layer_size = net->layer_size[net->num_layer-1];        //output size
    int recog = 0;
    // init weight with bias with random values
    for (i = 0; i < TOTAL_WEIGHTS(net); i++) {
        net->weight[i] = (float)rand()/(RAND_MAX/2)-1;
    }

    for (i = 0; i < TOTAL_NEURONS(net); i++) {
        net->bias[i] = 0;
    }
     for (i = 0; i < net->epoch; i++)
     {
        for (j = 0; j < nr_loop; j++)//j는 업데이트 하는 번수 (전체데이터를  mini batch로 나눈 값)
        {   
            // copy input and output for SGD
            for (k = 0; k < net->mini_batch_size; k++)
            {                   //k는데이터 번호를 뜻합니다, mini batch 사이즈 전까지 증가합니다
                int s_index = (int) rand()%nr_train;
                load_kth_data(net, s_index, k);
            }
      // feedforward + feedbackward      mini_batch size 만큼 다하고 함수들 실행
            feedforward(net);
            cost_report(net);
            feedbackward(net);
            update_parameters(net);
        }
        net->cost_rate = (net->cost_rate)/((net->mini_batch_size)*nr_loop);
        //reporting cost
        printf("%dth epoch  cost = %lf  \n", i,net->cost_rate);
        net->cost_rate = 0;
    }
        // test per every epoch
        recog = predict(net);
        if(recog > net->best_recog)
            net->best_recog = recog;
        printf("result  %d / %d \n", recog, net->nr_test_data);
}


void feedforward(struct network *net)
{
    timeutils *t_feedforward = &net->t_feedforward;
    
    START_TIME(t_feedforward);
    //printf("forward mode %d\n", net->mode[0]);

    switch (net->mode[0]) {
        case 0:
            feedforward00(net);
            break;
        case 1:
            feedforward01(net);
            break;
        case 2:
            feedforward02(net);
            break;
        case 3:
            feedforward03(net);
            break;
    }
    END_TIME(t_feedforward);
}

void feedforward01(struct network *net)
{
    int i, j, k, l, m;
    float sum = 0.0;

    for (i = 0; i < net->num_layer-1; i++)
    {
        #pragma omp parallel for num_threads(net->thread[0]) private(j, k, l) reduction(+:sum) collapse(2)
        for(j=0;j<net->mini_batch_size;j++)
        {
            for (k = 0; k < net->layer_size[i+1]; k++)
            {
                for (l = 0; l < net->layer_size[i]; l++)
                {
                    sum = sum + NEURON(net, i, j, l) * WEIGHT(net, i, l, k);
                }
                ZS(net, i+1, j, k) = sum + BIAS(net, i+1, k);
                NEURON(net, i+1, j, k) = sigmoid(ZS(net, i+1, j, k));
                sum = 0.0;
            }
        }
    }
}

void feedforward02(struct network *net)
{
    int per_thread_data=net->mini_batch_size/net->thread[0];
// vmode I :        
    #pragma omp parallel for num_threads(net->thread[0]) 
    for(int t=0;t<net->mini_batch_size; t+= per_thread_data){
        __m512 v1, v2, vd, bias;
        __m128 v128;
        for(int j=t;j<t+per_thread_data;j+=VECTOR_LEN) {
            for (int i = 0; i < net->num_layer-1; i++) {               
                for (int l = 0; l < net->layer_size[i]; l++) {
                    v1 = _mm512_load_ps(&NEURON(net,i,j,l));
                    for (int k = 0; k < net->layer_size[i+1]; k++) {    
                        v128 = _mm_broadcast_ss (&WEIGHT(net,i,l,k));
                        v2 = _mm512_broadcastss_ps(v128);
                        if (l==0)
                            vd = _mm512_mul_ps(v1, v2);
                        else{
                            vd = _mm512_load_ps(&NEURON(net,i+1,j,k));
                            vd = _mm512_fmadd_ps(v1, v2, vd);
                        }
                        if (l==net->layer_size[i]-1){
                            v128 = _mm_broadcast_ss (&BIAS(net,i+1,k));
                            bias = _mm512_broadcastss_ps(v128);
                            vd = _mm512_add_ps(vd, bias);
                            _mm512_store_ps(&ZS(net,i+1,j,k), vd);
                            vd = Vsigmoid(vd);
                        }   
                        _mm512_store_ps(&NEURON(net,i+1,j,k), vd);
                    }
                }
            }
        }
    }
}       

void feedforward03(struct network *net)
{
    int per_thread_data=net->mini_batch_size/net->thread[0];
// vmode II :       
    #pragma omp parallel for num_threads(net->thread[0]) 
    for(int t=0;t<net->mini_batch_size; t+= per_thread_data){
        __m512 v1, v2, vd, bias;
        __m128 v128;
        for(int j=t;j<t+per_thread_data;j+=VECTOR_LEN) {
            for (int i = 0; i < net->num_layer-1; i++) {               
                for (int k = 0; k < net->layer_size[i+1]; k++) {    
                    vd = _mm512_setzero_ps();
                    for (int l = 0; l < net->layer_size[i]; l++) {
                        v1 = _mm512_load_ps(&NEURON(net,i,j,l));
                        v128 = _mm_broadcast_ss (&WEIGHT(net,i,l,k));
                        v2 = _mm512_broadcastss_ps(v128);
                        vd = _mm512_fmadd_ps(v1, v2, vd);
                    }
                    v128 = _mm_broadcast_ss (&BIAS(net,i+1,k));
                    bias = _mm512_broadcastss_ps(v128);
                    vd = _mm512_add_ps(vd, bias);
                    _mm512_store_ps(&ZS(net,i+1,j,k), vd);
                    vd = Vsigmoid(vd);
                    _mm512_store_ps(&NEURON(net,i+1,j,k), vd);
                }
            }
        }
    }
}

void feedforward00(struct network *net)
{
    float *tmp, *tmp_bias;
    int i, j, k;
    for (i = 0; i < net->num_layer-1; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, net->mini_batch_size, net->layer_size[i+1], net->layer_size[i], 1.0, (const float *)&NEURON(net, i, 0, 0),net->layer_size[i], (const float *)&WEIGHT(net, i, 0, 0), net->layer_size[i+1], 0.0,&NEURON(net, i+1, 0, 0), net->layer_size[i+1]); //weight 와 입력값을 곱해서 배열에 저장합니다.
        #pragma omp parallel for num_threads(net->thread[0]) 
        for (j = 0; j < net->mini_batch_size; j++)
            for (int k = 0; k < net->layer_size[i+1]; k++) { 
                ZS(net, i+1, j, k) = NEURON(net,i+1,j,k)+ BIAS(net,i+1,k);
                NEURON(net, i+1, j, k) = sigmoid(ZS(net, i+1, j, k)); //zs에  sigmoid를 취한 값을 그다음 뉴런에 저장합니다!!
            }
    }
}

void feedbackward(struct network *net)
{
    timeutils *t_feedbackward = &net->t_feedbackward;
    
    START_TIME(t_feedbackward);
    switch (net->mode[1]) {
        case 0:
            feedbackward00(net);
            break;
        case 1:
            feedbackward01(net);
            break;
        case 2:
            feedbackward02(net);
            break;
        case 3:
            feedbackward03(net);
            break;
    }
    END_TIME(t_feedbackward);
}

void feedbackward01(struct network *net)
{
    int i, j, k, l;
    float sum = 0.0;

    #pragma omp parallel for num_threads(net->thread[1]) private(i, j) collapse(2)
    for (i = 0; i < net->mini_batch_size; i++) {
        for (j = 0; j < net->layer_size[net->num_layer-1]; j++) {
            //  calculate delta in last output layer
            ERROR(net, net->num_layer-1, i, j) =
            (NEURON(net, net->num_layer-1, i, j)-ERROR(net, net->num_layer-1, i, j)) *
            sigmoid_prime(ZS(net, net->num_layer-1, i, j));
        }
    }

    sum = 0.0;
    for (i = net->num_layer-2; i > 0; i--) {
    #pragma omp parallel for num_threads(net->thread[2]) private(j, k, l) reduction(+:sum) collapse(2)
        for (j = 0; j < net->mini_batch_size; j++) {
            for (k = 0; k < net->layer_size[i]; k++) {
                for (l = 0; l < net->layer_size[i+1]; l++) {
                    //  calculate delta from before layer
                    sum = sum + ERROR(net, i+1, j, l) * WEIGHT(net, i, k, l);
                }
                ERROR(net, i, j, k) = sum * sigmoid_prime(ZS(net, i, j, k));
                sum = 0.0;
            }
        }
    }
}

void feedbackward02(struct network *net)
{
    int t;
    int per_thread_data=net->mini_batch_size/net->thread[1];
// vmode I :        
    #pragma omp parallel for num_threads(net->thread[1]) 
    for(t=0;t<net->mini_batch_size; t+= per_thread_data){
        __m512 v1, v2, vd, bias;
        __m128 v128;
        for(int j=t;j<t+per_thread_data;j+=VECTOR_LEN) {
            for (int k = 0; k < net->layer_size[net->num_layer-1]; k++) {    
            //  calculate delta in last output layer
                v1 = _mm512_load_ps(&NEURON(net,net->num_layer-1,j,k));
                v2 = _mm512_load_ps(&ERROR(net,net->num_layer-1,j,k));
                v1 = _mm512_sub_ps(v1, v2);
                v2 = _mm512_load_ps(&ZS(net,net->num_layer-1,j,k));
                v2 = Vsigmoid_prime(v2);
                v2 = _mm512_mul_ps(v1, v2);
                _mm512_store_ps(&ERROR(net,net->num_layer-1,j,k), v2);
            }
        }
        for (int i = net->num_layer-2; i > 0; i--) {
            for(int j=t;j<t+per_thread_data;j+=VECTOR_LEN) {
                for (int k = 0; k < net->layer_size[i]; k++) {
                    vd = _mm512_setzero_ps();
                    for (int l = 0; l < net->layer_size[i+1]; l++) {
                    //  calculate delta from before layer
                        v1 = _mm512_load_ps(&ERROR(net,i+1,j,l));
                        v128 = _mm_broadcast_ss (&WEIGHT(net,i,k,l));
                        v2 = _mm512_broadcastss_ps(v128);
                        vd = _mm512_fmadd_ps(v1, v2, vd);
                    }
                    v2 = _mm512_load_ps(&ZS(net,i,j,k));
                    v2 = Vsigmoid_prime(v2);
                    vd = _mm512_mul_ps(vd, v2);
                    _mm512_store_ps(&ERROR(net,i,j,k), vd);
                }
            }
        }
    }
}

void feedbackward03(struct network *net)
{
    int t;
    int per_thread_data=net->mini_batch_size/net->thread[1];
// vmode II :       
    #pragma omp parallel for num_threads(net->thread[1]) 
    for(t=0;t<net->mini_batch_size; t+= per_thread_data){
        __m512 v1, v2, vd, bias;
        __m128 v128;
        for(int j=t;j<t+per_thread_data;j+=VECTOR_LEN) {
            for (int k = 0; k < net->layer_size[net->num_layer-1]; k++) {    
            //  calculate delta in last output layer
                v1 = _mm512_load_ps(&NEURON(net,net->num_layer-1,j,k));
                v2 = _mm512_load_ps(&ERROR(net,net->num_layer-1,j,k));
                v1 = _mm512_sub_ps(v1, v2);
                v2 = _mm512_load_ps(&ZS(net,net->num_layer-1,j,k));
                v2 = Vsigmoid_prime(v2);
                v2 = _mm512_mul_ps(v1, v2);
                _mm512_store_ps(&ERROR(net,net->num_layer-1,j,k), v2);
            }
        }
        for (int i = net->num_layer-2; i > 0; i--) {
            for(int j=t;j<t+per_thread_data;j+=VECTOR_LEN) {
                for (int l = 0; l < net->layer_size[i+1]; l++) {
                    v1 = _mm512_load_ps(&ERROR(net,i+1,j,l));
                    for (int k = 0; k < net->layer_size[i]; k++) {
                    //  calculate delta from before layer
                        v128 = _mm_broadcast_ss (&WEIGHT(net,i,k,l));
                        v2 = _mm512_broadcastss_ps(v128);
                        if (l==0)
                            vd = _mm512_mul_ps(v1, v2);
                        else {
                            vd = _mm512_load_ps(&ERROR(net,i,j,k));
                            vd = _mm512_fmadd_ps(v1, v2, vd);
                        }   
                        if (l==net->layer_size[i+1]-1){
                            v2 = _mm512_load_ps(&ZS(net,i,j,k));
                            v2 = Vsigmoid_prime(v2);
                            vd = _mm512_mul_ps(vd, v2);
                            _mm512_store_ps(&ERROR(net,i,j,k), vd);
                        }
                    }
                }
            }
        }
    }
}

void feedbackward00(struct network *net)
{
    int i, j, k, l;
    float * temp1;//neuron - error
    float * temp2;//sigmoid zs
    float * temp_error;

    temp1 = (float*)malloc(sizeof(float) * net->mini_batch_size * net->layer_size[net->num_layer-1]);
    temp2 = (float*)malloc(sizeof(float) * net->mini_batch_size * net->layer_size[net->num_layer-1]);

    // neuron - error
    vsSub(net->layer_size[net->num_layer-1]*net->mini_batch_size,&NEURON(net, net->num_layer-1, 0, 0),&ERROR(net, net->num_layer-1, 0, 0),temp1);

    //sigmoid zs
    #pragma omp parallel for num_threads(net->thread[1])
    for (i = 0; i < net->mini_batch_size*net->layer_size[net->num_layer-1]; i++)
    {
        temp2[i]=sigmoid_prime(ZS(net, net->num_layer-1, 0, i));
    }

    //temp1 * temp2 (when this loop is end  first delta is done!!)
    vsMul(net->layer_size[net->num_layer-1]*net->mini_batch_size,temp1,temp2,&ERROR(net, net->num_layer-1, 0, 0));

    //caculrate delta to using backpropagation algorithm
    for (i = net->num_layer-2; i > 0; i--)
    {
        for (j = 0; j < net->mini_batch_size; j++)
        {
            //temp_error = weight * past_error
            temp_error = (float*)malloc(sizeof(float)*net->layer_size[i]);

            //calculate temp_error
            cblas_sgemv (CblasRowMajor, CblasNoTrans,  net->layer_size[i], net->layer_size[i+1], 1.0,(const float *)&WEIGHT(net, i, 0, 0), net->layer_size[i+1],(const float *)&ERROR(net,i+1, j, 0),1 ,0.0 , temp_error , 1);

            //calculate delta = past error * weight * sigmoidprime(zs)
            #pragma omp parallel for num_threads(net->thread[2])
            for(k=0;k<net->layer_size[i];k++)
            {
                ERROR(net, i, j, k) = temp_error[k]*sigmoid_prime(ZS(net, i, j, k));
            }
        }
    }
}

void update_parameters(struct network *net)
{
    timeutils *t_update_parameters = &net->t_update_parameters;
    
    START_TIME(t_update_parameters);
    switch (net->mode[2]) {
        case 0:
            update_parameters00(net);
            break;
        case 1:
            update_parameters01(net);
            break;
        case 2:
            update_parameters02(net);
            break;
        case 3:
            update_parameters03(net);
            break;
    }
    END_TIME(t_update_parameters);
}



/* weight update : vectorization */
void update_parameters02(struct network *net)
{
    float eta = net->learning_rate;
    float mini = (float) net->mini_batch_size;

    //update bias
//  #pragma omp parallel for num_threads(net->thread[3]) collapse(2) 
    for (int i = 1; i < net->num_layer; i++) 
    #pragma omp parallel for num_threads(net->thread[3]) 
        for (int j = 0; j < net->layer_size[i]; j++) {
            float sum = 0.0;
            for (int k = 0; k < net->mini_batch_size; k+=VECTOR_LEN) {
                __m512 v1 = _mm512_load_ps(&ERROR(net, i, k, j));
                float vsum = _mm512_reduce_add_ps(v1);
                sum += vsum;
            }
            BIAS(net, i, j) -= (eta/mini)*sum;
        }
    // update weight

//  #pragma omp parallel for num_threads(net->thread[4]) collapse(3)
    for (int i = 0; i < net->num_layer-1; i++) {
    #pragma omp parallel for num_threads(net->thread[4]) collapse(2)
        for (int j = 0; j < net->layer_size[i]; j++) {
            for (int k = 0; k < net->layer_size[i+1]; k++) {
                float sum = 0.0;
                for (int l = 0; l < net->mini_batch_size; l+=VECTOR_LEN) {
                    __m512 v1 = _mm512_load_ps(&NEURON(net, i, l, j));
                    __m512 v2 = _mm512_load_ps(&ERROR(net, i+1, l, k));
                    v1 = _mm512_mul_ps(v1, v2);
                    float vsum = _mm512_reduce_add_ps(v1);
                    sum += vsum;
                }
                WEIGHT(net, i, j, k) -= (eta/mini)*sum;
            }
        }
    }
}

/* weight update : vectorization */
void update_parameters03(struct network *net)
{
    float eta = net->learning_rate;
    float mini = (float) net->mini_batch_size;
    int max_layer_size = net->layer_size[net->num_layer-1];
    //update bias
    #pragma omp parallel for num_threads(net->thread[3]) collapse(2) 
    for (int i = 1; i < net->num_layer; i++) 
        for (int j = 0; j < max_layer_size; j++) {
            if (j < net->layer_size[i]) {
            float sum = 0.0;
            for (int k = 0; k < net->mini_batch_size; k+=VECTOR_LEN) {
                __m512 v1 = _mm512_load_ps(&ERROR(net, i, k, j));
                float vsum = _mm512_reduce_add_ps(v1);
                sum += vsum;
            }
            BIAS(net, i, j) -= (eta/mini)*sum;
            }
        }
    // update weight

//  #pragma omp parallel for num_threads(net->thread[4]) collapse(3)
    for (int i = 0; i < net->num_layer-1; i++) {
    #pragma omp parallel for num_threads(net->thread[4]) collapse(2)
        for (int j = 0; j < net->layer_size[i]; j++) {
            for (int k = 0; k < net->layer_size[i+1]; k++) {
                float sum = 0.0;
                for (int l = 0; l < net->mini_batch_size; l+=VECTOR_LEN) {
                    __m512 v1 = _mm512_load_ps(&NEURON(net, i, l, j));
                    __m512 v2 = _mm512_load_ps(&ERROR(net, i+1, l, k));
                    v1 = _mm512_mul_ps(v1, v2);
                    float vsum = _mm512_reduce_add_ps(v1);
                    sum += vsum;
                }
                WEIGHT(net, i, j, k) -= (eta/mini)*sum;
            }
        }
    }
}

/* weight update : org */
void update_parameters01(struct network *net)
{
    int i, j, k, l;
    float eta = net->learning_rate;
    float mini = (float) net->mini_batch_size;
    float sum = 0;


    //update bias
    for (i = 1; i < net->num_layer; i++) 
    {
    #pragma omp parallel for num_threads(net->thread[3]) private(j, k, l)
        for (j = 0; j < net->layer_size[i]; j++)
        {
            for (k = 0; k < net->mini_batch_size; k++) 
            {
                BIAS(net, i, j) -= (eta/mini)*ERROR(net, i, k, j);
            }
        }
    }
    //update weight
    for (i = 0; i < net->num_layer-1; i++) {
    #pragma omp parallel for num_threads(net->thread[4]) private(j, k, l) collapse(2)
        for (j = 0; j < net->layer_size[i]; j++) {
            for (k = 0; k < net->layer_size[i+1]; k++) {
               #pragma omp simd reduction(+:sum) 
                for (l = 0; l < net->mini_batch_size; l++) {
                    //  calculate delta from before layer
                  sum  += (eta/mini)*(NEURON(net, i, l, j) * ERROR(net, i+1, l, k));
                }
                WEIGHT(net, i, j, k) -=sum;
                sum = 0;
            }
        }
    }
}

void update_parameters00(struct network *net)
{
    int i, j, k, l;
    float eta = net->learning_rate;
    float mini = (float) net->mini_batch_size;

	//update bias
	for (i = 1; i < net->num_layer; i++) 
	{
	#pragma omp parallel for num_threads(net->thread[3]) private(j, k)
		for (j = 0; j < net->layer_size[i]; j++)
		{
			for (k = 0; k < net->mini_batch_size; k++) 
			{
                BIAS(net, i, j) -= (eta/mini)*ERROR(net, i, k, j);
			}
		}
	}
    for (i = 0; i < net->num_layer-1; i++)
    {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,net->layer_size[i], net->layer_size[i+1],net->mini_batch_size, -(eta/mini), (const float *)&NEURON(net, i, 0, 0),net->layer_size[i], (const float *)&ERROR(net, i+1, 0, 0), net->layer_size[i+1], 1.0,&WEIGHT(net, i, 0, 0), net->layer_size[i+1]);
    }
}


int predict(struct network *net)
{
    int nr_true = 0;

    int i, j, k, l;
    float sum = 0.0;
    int nr_loop = (int)(net->nr_test_data);
    int first_layer_size = AC_NEURONS(net, 0);
    int last_layer_size = net->layer_size[net->num_layer-1];
    float cost_rate = 0;

    for (i = 0; i < nr_loop; i++) {
        // copy input to first layer of neuron array
        for (j = 0; j < first_layer_size; j++) {
            NEURON(net, 0, 0, j) = DATA_TEST_Q(net, i, j);
        }

        //feedforward
        sum = 0.0;
        for (j = 0; j < net->num_layer-1; j++) {
            #pragma omp parallel for num_threads(100) private(k, l) reduction(+:sum)
            for (k = 0; k < net->layer_size[j+1]; k++) {
                for (l = 0; l < net->layer_size[j]; l++) {
                    sum = sum + NEURON(net, j, 0, l) * WEIGHT(net, j, l, k);
                }

                ZS(net, j+1, 0, k) = sum + BIAS(net, j+1, k);
                NEURON(net, j+1, 0, k) = sigmoid(ZS(net, j+1, 0, k));
                sum = 0.0;
            }
        }

        float max = NEURON(net, net->num_layer-1, 0, 0);
        int max_idx = 0;

        for (j = 0; j < last_layer_size; j++) {
            if (NEURON(net, net->num_layer-1, 0, j) > max) {
                max = NEURON(net, net->num_layer-1, 0, j);
                max_idx = j;
            }
        }

        if (DATA_TEST_A(net, i) == max_idx)
            nr_true ++;
        
    }
    return nr_true;
}

void cost_report(struct network * net )
{
    int nr_train_data = net->nr_train_data;
    int i;
    for(i=0;i<net->mini_batch_size;i++)
    {
        net->cost_rate += 1- NEURON(net, net->num_layer-1,i,DATA_TRAIN_A(net,net->record_random[i]));
    }
}

void report(struct network *net)
{
    int *thread = (int *)net-> thread;
    int *mode = (int *)net->mode;

    timeutils *t_feedforward = &net->t_feedforward;
    timeutils *t_feedbackward = &net->t_feedbackward;
    timeutils *t_update_parameters = &net->t_update_parameters;
    timeutils t;
    timeutils *total = &t;

    TIMER_INIT(total);

    char *modeid[10] = {"MKL","OpenMP","Vector-mode1", "Vector-mode2"};
    int i = 0;
    FILE *f = fopen(net->report_file, "a+");
    
    fprintf( f, "\n=======================REPORT=======================\n");
    fprintf( f, "epoch : %d\n", net->epoch);
    fprintf( f, "learning_rate : %f\n", net->learning_rate);
    fprintf( f, "recognization rate : %d/%d\n", net->best_recog, net->nr_test_data);
    fprintf( f, "update_parameters thread1 : %d\n", thread[3]);
    fprintf( f, "update_parameters thread2 : %d\n", thread[4]);
    fprintf( f, "========================MODE========================\n");
    fprintf( f, "feedforward mode : %s\n", modeid[mode[0]]);
    fprintf( f, "feedbackward mode : %s\n", modeid[mode[1]]);
    fprintf( f, "update_parameters mode : %s\n", modeid[mode[2]]);
    fprintf( f, "========================TIME========================\n");
    fprintf( f, "feedforward : %ld.%d sec\n", TOTAL_SEC_TIME(t_feedforward), TOTAL_SEC_UTIME(t_feedforward));
    fprintf( f, "feedbackward : %ld.%d sec\n", TOTAL_SEC_TIME(t_feedbackward), TOTAL_SEC_UTIME(t_feedbackward));
    fprintf( f, "update_parameters : %ld.%d sec\n", TOTAL_SEC_TIME(t_update_parameters), TOTAL_SEC_UTIME(t_update_parameters));

    TIMER_ADD(t_feedforward, total);
    TIMER_ADD(t_feedbackward, total);
    TIMER_ADD(t_update_parameters, total);
    fprintf( f, "total : %ld.%d sec\n", TOTAL_SEC_TIME(total), TOTAL_SEC_UTIME(total));

}

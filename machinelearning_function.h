int predict(struct network *net);
double sigmoid_prime(double z);
double sigmoid(double z);
void train(struct network *net);
void init(struct network *net);
double randn(void);
void feedforward(struct network *net);
void back_pass(struct network *net);
void backpropagation(struct network *net);
void cost_report(struct network * net );
void report(struct network *net);

int layersize[NUM_LAYER] = {INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE};


double randn(void)
{
	    double v1, v2, s;

		    do {
				  v1 =  2 * ((double) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지
				  v2 =  2 * ((double) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
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
	timeutils *t_back_pass = &net->t_back_pass;
    timeutils *t_backpropagation = &net->t_backpropagation;

	net->best_recog = 0.0;
	TIMER_INIT(t_feedforward); //시간 초기화
    TIMER_INIT(t_back_pass);
	TIMER_INIT(t_backpropagation);
	
	net->cost_rate = 0;

	net->num_layer = NUM_LAYER;
	net->layer_size = (int *)layersize;

	net->learning_rate = LEARNING_RATE;	
	net->mini_batch_size = MINI_BATCH_SIZE;
	net->epoch = EPOCH;

	net->ac_weight = (int *) malloc(sizeof(double) * net->num_layer);
	net->ac_neuron = (int *) malloc(sizeof(double) * net->num_layer);

	net->thread = (int *)malloc(sizeof(int) * THREAD_MODE_NUM);
	net->mode = (int *)malloc(sizeof(int)*MODE_NUM);

	net->record_random = (int *)malloc(sizeof(int) * MINI_BATCH_SIZE);
	
	net->train_q_name = TRAIN_Q;
	net->train_a_name = TRAIN_A;
	net->test_q_name = TEST_Q;
	net->test_a_name = TEST_A;
	net->report_file = REPORT_F;

	//init mode & thread
	for(i=0;i<THREAD_MODE_NUM;i++)
		net->thread[i] =THREAD_NUM;
	
	for(i=0;i<MODE_NUM;i++)
	{	net->mode[i] =0;
		if(i==1)net->mode[i] = 1;
	}
	for (i = 0; i < net->num_layer; i++) {
		net->ac_neuron[i] = net->layer_size[i] + before_ac_neurals;//ac_neuron은 여태 누적한 neuron갯수..
		before_ac_neurals = net->ac_neuron[i];

		if (i == net->num_layer-1)
			continue;

		net->ac_weight[i] = net->layer_size[i] * net->layer_size[i+1] + before_ac_weights; //ac_weight는 여태 누적한 weight 의 갯수..
		before_ac_weights = net->ac_weight[i]; 
	}

	net->neuron = (double *) malloc(sizeof(double) * net->mini_batch_size * TOTAL_NEURONS(net)); //neuron 배열의 크기는 minibatch_size * 총 뉴련의 숫자
	net->zs = (double *) malloc(sizeof(double) * net->mini_batch_size * TOTAL_NEURONS(net));
	net->error =  (double *) malloc(sizeof(double) * net->mini_batch_size * TOTAL_NEURONS(net));
	net->bias = (double *) malloc(sizeof(double) * TOTAL_NEURONS(net));
	net->weight = (double *) malloc(sizeof(double) * TOTAL_WEIGHTS(net));


	for (i = 0; i < TOTAL_WEIGHTS(net); i++) {
        net->weight[i] = randn();
	}

	for (i = 0; i < TOTAL_NEURONS(net); i++) {
        net->bias[i] = randn();
	}
}
#if 0
void train(struct network *net)
{

	int i, j, k, l;
	int nr_train = net->nr_train_data;
	int nr_loop = (int)(net->nr_train_data/net->mini_batch_size);   //전체데이터를 미니배치 사이즈 만큼 나눈 수 입니다.(업데이트 할 숫자)
	int first_layer_size = AC_NEURONS(net, 0);//input  size
	int last_layer_size = net->layer_size[net->num_layer-1];     	//output size
	int recog = 0;
	// init weight with bias with random values
	for (i = 0; i < TOTAL_WEIGHTS(net); i++) {
        net->weight[i] = (double)rand()/(RAND_MAX/2)-1;
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
				// copy input to first layer of neuron array
				for (l = 0; l < first_layer_size; l++)                    //l은 28*28 까지 증가합니다
					NEURON(net, 0, k, l) = DATA_TRAIN_Q(net, s_index, l); //s_index 번째 데이터를 가져옵니다 그것을 net->neuron[net->layer_size[0]*(k) + (l)] 에 넣습니다.
                                                                        //즉 neuron 배열에 차곡차곡 랜덤한 인풋값을 넣습니다.
               for (l = 0; l < last_layer_size; l++)
                    ERROR(net, net->num_layer-1, k, l) = 0.0;
				// copy output to error array
				ERROR(net, net->num_layer-1, k, DATA_TRAIN_A(net, s_index)) = 1.0; //답안 배열에 1의값 넣습니다.
				net->record_random[k] = s_index;
			}
            // feedforward + back_pass      mini_batch size 만큼 다하고 함수들 실행
            feedforward(net);
			cost_report(net);
            back_pass(net);
            backpropagation(net);
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
	int i, j, k, l, m;
	double sum = 0.0;
    timeutils *t_feedforward = &net->t_feedforward;
	
	// feedforward
	START_TIME(t_feedforward);
    sum = 0.0;
if(net->mode[0])
{	
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
else
{  
	double *tmp, *tmp_bias;

    for (i = 0; i < net->num_layer-1; i++) 
	{
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, net->mini_batch_size, net->layer_size[i+1], net->layer_size[i], 1.0, (const double *)&NEURON(net, i, 0, 0),net->layer_size[i], (const double *)&WEIGHT(net, i, 0, 0), net->layer_size[i+1], 0.0,&NEURON(net, i+1, 0, 0), net->layer_size[i+1]); //weight 와 입력값을 곱해서 배열에 저장합니다.

		#pragma omp parallel for num_threads(net->thread[0]) 
		for (j = 0; j < net->mini_batch_size; j++)
            for (k = 0; k < net->layer_size[i+1]; k++)
			{ 
				ZS(net, i+1, j, k) = NEURON(net,i+1,j,k)+ BIAS(net,i+1,k);
				NEURON(net, i+1, j, k) = sigmoid(ZS(net, i+1, j, k)); //zs에  sigmoid를 취한 값을 그다음 뉴런에 저장합니다!!
			}
	}
}
	END_TIME(t_feedforward);
}

void back_pass(struct network *net)
{
	int i, j, k, l;
    int nr_chunk = net->thread[0];
	double sum = 0.0;
    timeutils *t_back_pass = &net->t_back_pass;

	START_TIME(t_back_pass);

if(net->mode[1])
{
// calculate delta
	#pragma omp parallel for num_threads(net->thread[1]) private(i, j) collapse(2)
	for (i = 0; i < net->mini_batch_size; i++) {
		for (j = 0; j < net->layer_size[net->num_layer-1]; j++) {
			//	calculate delta in last output layer
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
					//	calculate delta from before layer
					sum = sum + ERROR(net, i+1, j, l) * WEIGHT(net, i, k, l);
				}
				ERROR(net, i, j, k) = sum * sigmoid_prime(ZS(net, i, j, k));
				sum = 0.0;
			}
		}
	}
}
else
{
	double * temp1;//neuron - error
    double * temp2;//sigmoid zs
    double * temp_error;

    temp1 = (double*)malloc(sizeof(double) * net->mini_batch_size * net->layer_size[net->num_layer-1]);
    temp2 = (double*)malloc(sizeof(double) * net->mini_batch_size * net->layer_size[net->num_layer-1]);

    // neuron - error
    vdSub(net->layer_size[net->num_layer-1]*net->mini_batch_size,&NEURON(net, net->num_layer-1, 0, 0),&ERROR(net, net->num_layer-1, 0, 0),temp1);

    //sigmoid zs
    #pragma omp parallel for num_threads(net->thread[1])
	    for (i = 0; i < net->mini_batch_size*net->layer_size[net->num_layer-1]; i++)
            {
		         temp2[i]=sigmoid_prime(ZS(net, net->num_layer-1, 0, i));
	    	}

    //temp1 * temp2 (when this loop is end  first delta is done!!)
    vdMul(net->layer_size[net->num_layer-1]*net->mini_batch_size,temp1,temp2,&ERROR(net, net->num_layer-1, 0, 0));

    //caculrate delta to using backpropagation algorithm
    for (i = net->num_layer-2; i > 0; i--)
    {
		for (j = 0; j < net->mini_batch_size; j++)
        {
            //temp_error = weight * past_error
            temp_error = (double*)malloc(sizeof(double)*net->layer_size[i]);

            //calculate temp_error
            cblas_dgemv (CblasRowMajor, CblasNoTrans,  net->layer_size[i], net->layer_size[i+1], 1.0,(const double *)&WEIGHT(net, i, 0, 0), net->layer_size[i+1],(const double *)&ERROR(net,i+1, j, 0),1 ,0.0 , temp_error , 1);

            //calculate delta = past error * weight * sigmoidprime(zs)
            #pragma omp parallel for num_threads(net->thread[2])
            for(k=0;k<net->layer_size[i];k++)
            {
                ERROR(net, i, j, k) = temp_error[k]*sigmoid_prime(ZS(net, i, j, k));
            }

        }
    }
}
	END_TIME(t_back_pass);
}

/* Operation like backpropagation */
void backpropagation(struct network *net)
{
	int i, j, k, l;
    timeutils *t_backpropagation = &net->t_backpropagation;
	double eta = net->learning_rate;
	double mini = (double) net->mini_batch_size;
	double sum = 0;


	START_TIME(t_backpropagation);
	
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

if(net->mode[2])
{	
	// update weight
	for (i = 0; i < net->num_layer-1; i++) {
#pragma omp parallel for num_threads(net->thread[4]) private(j, k, l) collapse(2)
		for (j = 0; j < net->layer_size[i]; j++) {
			for (k = 0; k < net->layer_size[i+1]; k++) {
               #pragma omp simd reduction(+:sum) 
				for (l = 0; l < net->mini_batch_size; l++) {
					//	calculate delta from before layer
                 sum  += (eta/mini)*(NEURON(net, i, l, j) * ERROR(net, i+1, l, k));
				}
WEIGHT(net, i, j, k) -=sum;
sum = 0;
			}
		}
	}
}
// update weight
else
{
	for (i = 0; i < net->num_layer-1; i++)
	{
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,net->layer_size[i], net->layer_size[i+1],net->mini_batch_size, -(eta/mini), (const double *)&NEURON(net, i, 0, 0),net->layer_size[i], (const double *)&ERROR(net, i+1, 0, 0), net->layer_size[i+1], 1.0,&WEIGHT(net, i, 0, 0), net->layer_size[i+1]);
	}
}

	END_TIME(t_backpropagation);
}

double sigmoid(double z)
{
	return (1/(1 + exp(-z)));
}

double sigmoid_prime(double z)
{
	return sigmoid(z)*(1-sigmoid(z));
}

int predict(struct network *net)
{
	int nr_true = 0;

	int i, j, k, l;
	double sum = 0.0;
	int nr_loop = (int)(net->nr_test_data);
	int first_layer_size = AC_NEURONS(net, 0);
	int last_layer_size = net->layer_size[net->num_layer-1];
	double cost_rate = 0;

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

		double max = NEURON(net, net->num_layer-1, 0, 0);
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
    timeutils *t_back_pass = &net->t_back_pass;
    timeutils *t_backpropagation = &net->t_backpropagation;
    timeutils t;
    timeutils *total = &t;

    TIMER_INIT(total);

	char *modeid[2] = {"MKL","OpenMP"};
	int i = 0;
	FILE *f = fopen(net->report_file, "a+");
	
	fprintf( f, "\n=======================REPORT=======================\n");
	fprintf( f, "epoch : %d\n", net->epoch);
	fprintf( f, "learning_rate : %f\n", net->learning_rate);
	fprintf( f, "recognization rate : %d/%d\n", net->best_recog, net->nr_test_data);
	fprintf( f, "=======================THREADS======================\n");
	fprintf( f, "feedforward thread : %d\n", thread[0]);
	fprintf( f, "back_pass thread1 : %d\n", thread[1]);
	fprintf( f, "back_pass thread2 : %d\n", thread[2]);
	fprintf( f, "backpropagation thread1 : %d\n", thread[3]);
	fprintf( f, "backpropagation thread2 : %d\n", thread[4]);
	fprintf( f, "========================MODE========================\n");
	fprintf( f, "feedforward mode : %s\n", modeid[mode[0]]);
	fprintf( f, "back_pass mode : %s\n", modeid[mode[1]]);
	fprintf( f, "backpropagation mode : %s\n", modeid[mode[2]]);
	fprintf( f, "========================TIME========================\n");
	fprintf( f, "feedforward : %ld.%d sec\n", TOTAL_SEC_TIME(t_feedforward), TOTAL_SEC_UTIME(t_feedforward));
	fprintf( f, "back_pass : %ld.%d sec\n", TOTAL_SEC_TIME(t_back_pass), TOTAL_SEC_UTIME(t_back_pass));
	fprintf( f, "backpropagation : %ld.%d sec\n", TOTAL_SEC_TIME(t_backpropagation), TOTAL_SEC_UTIME(t_backpropagation));

    TIMER_ADD(t_feedforward, total);
    TIMER_ADD(t_back_pass, total);
    TIMER_ADD(t_backpropagation, total);
	fprintf( f, "total : %ld.%d sec\n", TOTAL_SEC_TIME(total), TOTAL_SEC_UTIME(total));

}
#endif

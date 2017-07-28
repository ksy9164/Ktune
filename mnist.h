#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

#define TRAIN_DATA_NUM 60000
#define TEST_DATA_NUM 10000

#define TRAIN_Q "./data/train-images-idx3-ubyte"
#define TRAIN_A "./data/train-labels-idx1-ubyte"
#define TEST_Q "./data/t10k-images-idx3-ubyte"
#define TEST_A "./data/t10k-labels-idx1-ubyte"

void mnist_load(struct network * net)
{
	int i,j;

    FILE* inf_tr = fopen(TRAIN_Q, "rb");
    FILE* outf_tr = fopen(TRAIN_A, "rb");
	FILE* inf_te = fopen(TEST_Q, "rb");
	FILE* outf_te = fopen(TEST_A, "rb");
	
	int temp[16] = {0};

    net->nr_train_data = TRAIN_DATA_NUM;
	net->nr_test_data = TEST_DATA_NUM;

  	net->train_q = (double *) malloc(net->layer_size[0] * net->nr_train_data *sizeof(double));
	net->train_a = (int *)calloc(net->nr_train_data,sizeof(int));
    net->test_q = (double *) malloc(net->layer_size[0] * net->nr_test_data * sizeof(double));
	net->test_a = (int *) calloc(net->nr_test_data, sizeof(int));

    fread(temp, sizeof(unsigned char), 16, inf_tr);	// trash
    fread(temp, sizeof(unsigned char), 8, outf_tr);	// trash
    fread(temp, sizeof(unsigned char), 16, inf_te);	// trash
    fread(temp, sizeof(unsigned char), 8, outf_te);	// trash

  	unsigned char * temp_tr_q = (unsigned char *)malloc(net->layer_size[0] * net->nr_train_data *sizeof(unsigned int));
	unsigned char * temp_tr_a = (unsigned char *)calloc(net->nr_train_data,sizeof(unsigned int));
	unsigned char * temp_te_q = (unsigned char *)malloc(net->layer_size[0] * net->nr_test_data * sizeof(unsigned int));
	unsigned char * temp_te_a = (unsigned char *)calloc(net->nr_test_data, sizeof(unsigned int));
	
	fread(temp_tr_q,sizeof(unsigned char),INPUT_SIZE*TRAIN_DATA_NUM,inf_tr);
	fread(temp_tr_a,sizeof(unsigned char),TRAIN_DATA_NUM,outf_tr);
	fread(temp_te_q,sizeof(unsigned char),INPUT_SIZE*TRAIN_DATA_NUM,inf_te);
	fread(temp_te_a,sizeof(unsigned char),TRAIN_DATA_NUM,outf_te);
#if 0
	for(i=0;i<TRAIN_DATA_NUM;i++)
	{
		for(j=0;j<net->layer_size[0];j++)
		{
			net->train_q[i*net->layer_size[0]+j ]= (double)temp_tr_q[i*net->layer_size[0]+j];
		}
		net->train_a[i] = (int)temp_tr_a[i];
	}

	for(i=0;i<TEST_DATA_NUM;i++)
	{
		for(j=0;j<net->layer_size[0];j++)
		{
			net->test_q[i*net->layer_size[0]+j ]= (double)temp_te_q[i*net->layer_size[0]+j];
		}
		net->test_a[i] = (int)temp_te_a[i];
	}
#else
	for(i=0;i<TRAIN_DATA_NUM;i++)
	{
		for(j=0;j<net->layer_size[0];j++)
		{	if((double)temp_tr_q[i*net->layer_size[0]+j] == 0)
			{	
				net->train_q[i*net->layer_size[0]+j ] = 0;
			}
			
			else
			{
				net->train_q[i*net->layer_size[0]+j ] = 1;
			}
		}
		net->train_a[i] = (int)temp_tr_a[i];
	}

	for(i=0;i<TEST_DATA_NUM;i++)
	{
		for(j=0;j<net->layer_size[0];j++)
		{	
			if( (double)temp_te_q[i*net->layer_size[0]+j] == 0)
			{
				net->test_q[i*net->layer_size[0]+j ] = 0;
			}

			else
			{
				net->test_q[i*net->layer_size[0]+j ] = 1;
			}
		}
		net->test_a[i] = (int)temp_te_a[i];
	}
#endif
free(temp_tr_q);
free(temp_tr_a);
free(temp_te_q);
free(temp_te_a);
/*
 for(int k=12000;k<12012;k++)
{
	for(i=0;i<28;i++)
	{
		for(j=0;j<28;j++)
		{
		if(net->train_q[(i*28+j)+k*784] == 0) printf("1 ");
		else printf("0 ");
		}
	printf("\n");
	}
	printf("\n answer = %d \n\n",net->train_a[k]);
}
*/
}

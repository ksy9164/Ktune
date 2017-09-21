#define INPUT_SIZE 3072
#define OUTPUT_SIZE 10

#define TRAIN_DATA_NUM 50000
#define TEST_DATA_NUM 10000

#define TRAIN_1 "./data/data_batch_1.bin"
#define TRAIN_2 "./data/data_batch_2.bin"
#define TRAIN_3 "./data/data_batch_3.bin"
#define TRAIN_4 "./data/data_batch_4.bin"
#define TRAIN_5 "./data/data_batch_5.bin"
#define TEST "./data/test_batch.bin"

//#define TRAIN_1 "./data/data_batch_1.bin"
//#define TRAIN_2 "./data/data_batch_1.bin"
//#define TRAIN_3 "./data/data_batch_1.bin"
//#define TRAIN_4 "./data/data_batch_1.bin"
//#define TRAIN_5 "./data/data_batch_1.bin"
//#define TEST "./data/data_batch_1.bin"
void cifar_load(struct network * net)
{
	int cnt,i,j;

    FILE* inf_tr1 = fopen(TRAIN_1, "rb");
    FILE* inf_tr2 = fopen(TRAIN_2, "rb");
    FILE* inf_tr3 = fopen(TRAIN_3, "rb");
    FILE* inf_tr4 = fopen(TRAIN_4, "rb");
    FILE* inf_tr5 = fopen(TRAIN_5, "rb");
	FILE* inf_te = fopen(TEST, "rb");

    net->nr_train_data = TRAIN_DATA_NUM;
	net->nr_test_data = TEST_DATA_NUM;

    //진짜로 데이터를 담을 그릇!!
  	net->train_q = (double *) malloc(net->layer_size[0] * net->nr_train_data *sizeof(double));
	net->train_a = (int *)calloc(net->nr_train_data,sizeof(int));
    net->test_q = (double *) malloc(net->layer_size[0] * net->nr_test_data * sizeof(double));
	net->test_a = (int *) calloc(net->nr_test_data, sizeof(int));

    //변환을 위한 임시 데이터 저장소
  	unsigned char * temp_tr1 = (unsigned char *)malloc((net->layer_size[0]+1) * net->nr_train_data/5*sizeof(unsigned char));
  	unsigned char * temp_tr2 = (unsigned char *)malloc((net->layer_size[0]+1) * net->nr_train_data/5*sizeof(unsigned char));
  	unsigned char * temp_tr3 = (unsigned char *)malloc((net->layer_size[0]+1) * net->nr_train_data/5*sizeof(unsigned char));
  	unsigned char * temp_tr4 = (unsigned char *)malloc((net->layer_size[0]+1) * net->nr_train_data/5*sizeof(unsigned char));
  	unsigned char * temp_tr5 = (unsigned char *)malloc((net->layer_size[0]+1) * net->nr_train_data/5*sizeof(unsigned char));
  	unsigned char * temp_te = (unsigned char *)malloc((net->layer_size[0]+1) * net->nr_test_data*sizeof(unsigned char));

	fread(temp_tr1,sizeof(unsigned char),(INPUT_SIZE+1)*(TRAIN_DATA_NUM/5),inf_tr1);
	fread(temp_tr2,sizeof(unsigned char),(INPUT_SIZE+1)*(TRAIN_DATA_NUM/5),inf_tr2);
	fread(temp_tr3,sizeof(unsigned char),(INPUT_SIZE+1)*(TRAIN_DATA_NUM/5),inf_tr3);
	fread(temp_tr4,sizeof(unsigned char),(INPUT_SIZE+1)*(TRAIN_DATA_NUM/5),inf_tr4);
	fread(temp_tr5,sizeof(unsigned char),(INPUT_SIZE+1)*(TRAIN_DATA_NUM/5),inf_tr5);
	fread(temp_te,sizeof(unsigned char),(INPUT_SIZE+1)*TRAIN_DATA_NUM,inf_te);

    //1st data write
    cnt=0;
    for(i=0;i<10000;i++)
    {
        for(j=0;j<(net->layer_size[0]+1);j++)
        {
            if(j==0)
            {
                net->train_a[i]= (int)temp_tr1[cnt*(net->layer_size[0]+1)];
                net->test_a[i]= (int)temp_te[cnt*(net->layer_size[0]+1)];
            }
            else
            {
                net->train_q[i*net->layer_size[0]+j-1]= (double)temp_tr1[cnt*(net->layer_size[0]+1)+j];
                net->test_q[i*net->layer_size[0]+j-1]= (double)temp_te[cnt*(net->layer_size[0]+1)+j];
            }
        }
        cnt++;
    }
    
    cnt=0;
    for(i=10000;i<20000;i++)
    {
        for(j=0;j<(net->layer_size[0]+1);j++)
        {
            if(j==0)
                net->train_a[i]= (int)temp_tr2[cnt*(net->layer_size[0]+1)];
            else
                net->train_q[i*net->layer_size[0]+j-1]= (double)temp_tr2[cnt*(net->layer_size[0]+1)+j];
        }
        cnt++;
    }
    
    cnt=0;
    for(i=20000;i<30000;i++)
    {
        for(j=0;j<(net->layer_size[0]+1);j++)
        {
            if(j==0)
                net->train_a[i]= (int)temp_tr3[cnt*(net->layer_size[0]+1)];
            else
                net->train_q[i*net->layer_size[0]+j-1]= (double)temp_tr3[cnt*(net->layer_size[0]+1)+j];
        }
        cnt++;
    }

    cnt=0;
    for(i=30000;i<40000;i++)
    {
        for(j=0;j<(net->layer_size[0]+1);j++)
        {
            if(j==0)
                net->train_a[i]= (int)temp_tr4[cnt*(net->layer_size[0]+1)];
            else
                net->train_q[i*net->layer_size[0]+j-1]= (double)temp_tr4[cnt*(net->layer_size[0]+1)+j];
        }
        cnt++;
    }

    cnt=0;
    for(i=40000;i<50000;i++)
    {
        for(j=0;j<(net->layer_size[0]+1);j++)
        {
            if(j==0)
                net->train_a[i]= (int)temp_tr5[cnt*(net->layer_size[0]+1)];
            else
                net->train_q[i*net->layer_size[0]+j-1]= (double)temp_tr5[cnt*(net->layer_size[0]+1)+j];
        }
        cnt++;
    }
}

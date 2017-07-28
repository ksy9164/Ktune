#include<boost/python.hpp>
#include<cstdarg>
#include<iostream>
#include"timeutils.h"
#include"network_definition.h"
#include"mnist.h"
#include"machinelearning_function.h"
using namespace std;

class Sequential
{
	private:
	struct network * net;
    int recog = 0;
	int i;
    int temp_cnt = 0;
	
    public:
    Sequential()
    {
        net = (struct network *)malloc(sizeof(struct network));
    }
    void network(char* str)
    {
        const char * network_name_temp = str;
        int i;
        for(i=0 ;i<NUMBER_OF_NETWORK+1 ;i++)
        {
             if(!strcmp(network_Name[i],network_name_temp))
             {
                net->network_info = i;
                break;
             }
        }
        if(i == NUMBER_OF_NETWORK+1)
            printf("its error network name does not matching \n");
        printf("%d %d",net->network_info,i);
    }
    void layersize(int num)
    {
        net->num_layer = num;
        net->layer_size = (int *)malloc(sizeof(int)*num);
    }
    void add(int num)
    {
        if(temp_cnt == net->num_layer)
            printf("its full  error!!\n");
        else
        {
            net->layer_size[temp_cnt] = num;
            temp_cnt ++; 
        }
    }
    void action(char* str)
    {
        const char * action_name_temp = str;
        int i;
        for(i=0 ;i<NUMBER_OF_ACTION+1 ;i++)
        {
             if(!strcmp(action_Name[i],action_name_temp))
             {
                net->action_info = i;
                break;
             }
        }
        if(i == NUMBER_OF_ACTION+1)
            printf("its error action name does not matching \n");
        printf("%d %d",net->action_info,i);
    }

    void loss(char* str)
    {
        const char * loss_name_temp = str;
        int i;
        for(i=0 ;i<NUMBER_OF_LOSS+1 ;i++)
        {
             if(!strcmp(loss_Name[i],loss_name_temp))
             {
                net->loss_info = i;
                break;
             }
        }
        if(i == NUMBER_OF_LOSS+1)
            printf("its error loss name does not matching \n");
        printf("%d %d",net->loss_info,i);
    }

    void optimizer(char* str)
    {
        const char * optimizer_name_temp = str;
        int i;
        for(i=0 ;i<NUMBER_OF_OPTIMIZER+1 ;i++)
        {
             if(!strcmp(optimizer_Name[i],optimizer_name_temp))
             {
                net->optimizer_info = i;
                break;
             }
        }
        if(i == NUMBER_OF_OPTIMIZER+1)
            printf("its error optimizer name does not matching \n");
        printf("%d %d",net->optimizer_info,i);
    }

    void data(char* str)
    {
        const char * data_name_temp = str;
        int i;
        for(i=0 ;i<NUMBER_OF_DATA+1 ;i++)
        {
             if(!strcmp(data_Name[i],data_name_temp))
             {
                net->data_info = i;
                break;
             }
        }
        if(i == NUMBER_OF_DATA+1)
            printf("its error data name does not matching \n");
        printf("%d %d",net->data_info,i);
    }
//    void add(int num , ...)
//    {
//        va_list list;
//        va_start(list,num);
//        net->num_layer = num;
//        net->layer_size = (int *)malloc(sizeof(int)*num);
//        for(int i=0;i<num;i++)
//        {
//            net->layer_size[i] = va_arg(list,int);
//        }
//        va_end(list);
//    }
    
	void help(void)
	{
		cout<<"here is the function of Ktune\n\n"<<endl;
		cout<<"1. network\n";
		cout<<"This is a function that determines what kind of network to use\n ";
		cout<<".networkd(""kind of network"") \n";
		cout<<"example : .network(""fully connected"") \n\n";

		cout<<"2. add\n";
		cout<<"This is a function that add layer\n ";
		cout<<".add(""number of layer , 1st layer number(input layer) , 2nd layer number ...., output layer number"") \n";
		cout<<"example : .add(""4, 784 , 120 , 40 , 10"") \n\n";

		cout<<"3. action\n";
		cout<<"This is a function that setting action function\n ";
		cout<<".action(""action function name"") \n";
		cout<<"example : .action(""sigmoid"") \n\n";


		cout<<"4. loss\n";
		cout<<"This is a function that setting loss function\n ";
		cout<<".loss(""loss function name"") \n";
		cout<<"example : .loss(""mean_squared_error"") \n\n";


		cout<<"5. optimizer\n";
		cout<<"This is a function that setting optimizer function\n ";
		cout<<".optimizer(""optimizer function name"") \n";
		cout<<"example : .optimizerz(""sgd"") \n\n";


		cout<<"6. data\n";
		cout<<"This is a function that setting determines what kind of data to use\n ";
		cout<<".data(""data name"") \n";
		cout<<"example : .data(""mnist"") \n\n";


		cout<<"7. fit\n";
		cout<<"This is a function that training data \n ";
		cout<<".fit(""EPOCH,Batch_size"") \n";
		cout<<"example : .fit(""5,100"") \n\n";
	}
/*
	void fit(void)
	{
		net = (struct network *) malloc (sizeof(struct network));
		init(net);
		mnist_load(net);
		train(net);
		report(net);
		free(net);
		return;
	}
*/	

};

using namespace boost::python;

BOOST_PYTHON_MODULE(Ktune)
{
    class_<Sequential>("Sequential")
          .def("help",&Sequential::help)
          .def("network",&Sequential::network)
          .def("layersize",&Sequential::layersize)
          .def("add",&Sequential::add)
          .def("action",&Sequential::action)
          .def("loss",&Sequential::loss)
          .def("optimizer",&Sequential::optimizer)
          .def("data",&Sequential::data)
//		  .def("fit",&Sequential::fit)
        ;   
};using namespace boost::python;


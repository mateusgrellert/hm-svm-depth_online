
#include "TEncSVM.h"
#include "TLibCommon/TComYuv.h"
#include "TLibCommon/TComPic.h"
#include <errno.h>


struct svm_parameter TEncSVM::param;		// set by parse_command_line
struct svm_problem TEncSVM::prob;		// set by read_problem
struct svm_node* TEncSVM::x_space;
char * TEncSVM::line;
int TEncSVM::max_line_len;

vector<int> TEncSVM::featureOrder[3];

struct svm_model* TEncSVM::model[NUM_SVM_MODES][4];
std::string TEncSVM::modelPath[NUM_SVM_MODES][4];
std::string TEncSVM::rangePath[NUM_SVM_MODES][4];
int TEncSVM::pictureCount;
std::ostringstream TEncSVM::feat_debug_str;
bool TEncSVM::enableSVM[NUM_SVM_MODES];
bool TEncSVM::encodeStarted;
bool TEncSVM::useScaling;
bool TEncSVM::structs_allocated;
double TEncSVM::splitTh[NUM_SVM_MODES][4];
struct svm_node* TEncSVM::svmNode;
double* TEncSVM::probEstimates;
int* TEncSVM::featIdx[NUM_SVM_MODES][4];
double* TEncSVM::featMin[NUM_SVM_MODES][4];
double* TEncSVM::featMax[NUM_SVM_MODES][4];
int TEncSVM::numFeatures[NUM_SVM_MODES][4];
int TEncSVM::labelCount[NUM_SVM_MODES][4];
int TEncSVM::totalCalls[4];
int TEncSVM::enabled;
    

void TEncSVM::init(){

    int d,mode;
    int maxNumFeatures = -1;

    encodeStarted = false;
    enabled = 1;
    for(mode = 0; mode < NUM_SVM_MODES; mode++){
        if(!enableSVM[mode])
            continue;
        for (d=0; d < 4; d++){
            if(mode == MODE_CU and d == 3)
                continue;
            if(!structs_allocated)
                featIdx[mode][d] = (int*) malloc(sizeof(int)*(N_FEATURES+1)); 
            labelCount[mode][d] = 0;
            totalCalls[d] = 0;
            //model[mode][d] = (struct svm_model*) malloc(sizeof(struct svm_model));
            model[mode][d] = svm_load_model(modelPath[mode][d].c_str(), featIdx[mode][d], N_FEATURES);


            numFeatures[mode][d] = featIdx[mode][d][N_FEATURES];
            if (numFeatures[mode][d] > maxNumFeatures)
                maxNumFeatures = numFeatures[mode][d];
        }
        

    }
                
    if(!structs_allocated){
        svmNode = (struct svm_node *) malloc((N_FEATURES+1)*sizeof(struct svm_node));
        probEstimates = (double*) malloc(sizeof(double)*3);
    }
    
    structs_allocated = true;
}





bool TEncSVM::parseRangeModel(std::string path, int mode, int depth){
    FILE *rangeFile = fopen(path.c_str(), "r");
    if (rangeFile == NULL){
        fprintf(stderr, "Error opening file %s\n", path.c_str());
        exit(1);
    }
    int idx;
    double max, min;
    
    bool fs = fscanf(rangeFile, "%*[^\n]\n"); //skip line
    fs = fscanf(rangeFile, "%*[^\n]\n"); //skip line
    
    while(!feof(rangeFile)){
        fs = fscanf(rangeFile, "%d %lf %lf", &idx, &min, &max);
        
        featMax[mode][depth][idx] = max;
        featMin[mode][depth][idx] = min;
    }
    
    fclose(rangeFile);
    return fs;
    
}

void TEncSVM::setCUFeatures(TComDataCU *&cu, int mode, int depth){

    feat_debug_str.str("");
    double val;
   // printf("Mode %d PU %d TrD %d\n", mode, puSize, trDepth);

    for(int i = 0; i < numFeatures[mode][depth]; i++){

        svmNode[i].index = featIdx[mode][depth][i]; // C_MODE
        val = TComDebug::getFeatureValue(cu, featIdx[mode][depth][i]+3); // use +3 to discard features that were not considered in the models
        if (useScaling)
            svmNode[i].value = scale(val, featIdx[mode][depth][i], mode, depth);
        else
            svmNode[i].value = val;
       feat_debug_str << svmNode[i].index << ":" << svmNode[i].value << " ";
    }
            
    svmNode[numFeatures[mode][depth]].index = -1; // last must be -1

    
   
    

    
}

//
//void TEncSVM::printSVMNode(int depth){
// for(int i = 0; i < numFeatures[depth]; i++){
//     std::cout << svmNode[i].index << ": " << svmNode[i].value << std::endl;
// }
//}

double TEncSVM::scale(double value, int index, int mode, int depth){
    if(value == featMin[mode][depth][index])
	value = 0;
    else if(value == featMax[mode][depth][index])
        value = 1;
    else
	value = (value-featMin[mode][depth][index])/
		(featMax[mode][depth][index]-featMin[mode][depth][index]);
    
    return value;
}

//void TEncSVM::updateMinMax(TComDataCU *&cu){
//      
//    int depth = cu->getDepth(0);
//
//    double val;
//    int index;
//    for(int i = 0; i < numFeatures[depth]; i++){
//
//        index = featIdx[depth][i]; // C_MODE
//        val = TComDebug::getFeatureValue( cu, index+3);
//        featMin[depth][index] = std::min(val,featMin[depth][index] );
//        featMax[depth][index] = std::max(val,featMax[depth][index] );
//    }
//
//
//    
//}

int TEncSVM::predictLabel(TComDataCU *&cu, int mode){
    
    int depth = cu->getDepth(0);
       
    

    if (splitTh[mode][depth] == 0){
        labelCount[mode][depth] ++;
        return 1;
    }
       
    
    if (splitTh[mode][depth] == 1){
        return 0;
    }
//    FILE* hmoutfile;
//    if (depth == 0)
//       hmoutfile = fopen("hmout_file_d0.libsvm","a");
//    else if (depth == 1)
//       hmoutfile = fopen("hmout_file_d1.libsvm","a");
//    else
//       hmoutfile = fopen("hmout_file_d2.libsvm","a");
       
       
    setCUFeatures(cu, mode, depth);

    int predict_label;
    predict_label =  model[mode][depth]->label[(int)svm_predict_probability(model[mode][depth],svmNode,probEstimates)];

    double prob[3];
    int i,numLabels;
    numLabels = NUM_LABELS[mode];
            

    for(i = 0; i < numLabels; i++){
        prob[model[mode][depth]->label[i]] = probEstimates[model[mode][depth]->label[i]];
    }
    


    predict_label = 0;
    if (prob[1] >= splitTh[mode][depth]){
        predict_label = 1;
        labelCount[mode][depth] ++;

    }
       

    return predict_label;

}


bool TEncSVM::isTrainingPic(){
 return (TEncSVM::pictureCount >= SVM_START_PIC)  &&\
        (((TEncSVM::pictureCount - SVM_START_PIC) % SVM_REFRESH_GOP) < SVM_TRAINING_SIZE ) ;
}

bool TEncSVM::trainSVM(){
    
 if(!enableSVM[MODE_CU]) return 0;
 
 if ((TEncSVM::pictureCount >= SVM_START_PIC)  &&\
    (((TEncSVM::pictureCount - SVM_START_PIC) % SVM_REFRESH_GOP) == SVM_TRAINING_SIZE ))
 {
    runSVMTrain();
    return 1;
 }
 else
    return 0;
}


void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}


void TEncSVM::read_problem(FILE *fp)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	char *endptr;
	char *idx, *val, *label;



	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
        
	while(readline(fp))
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;

	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;



	fclose(fp);
}


void TEncSVM::runSVMTrain(){
       
        fclose(TComDebug::libSVMFile[0]);
        fclose(TComDebug::libSVMFile[1]);
        fclose(TComDebug::libSVMFile[2]);
        //fclose(TComDebug::libSVMFile[3]);

        TComDebug::libSVMFile[0] = fopen("online_d0.libsvm","r");
        TComDebug::libSVMFile[1] = fopen("online_d1.libsvm","r");
        TComDebug::libSVMFile[2] = fopen("online_d2.libsvm","r");
        //TComDebug::libSVMFile[3] = fopen("online_d3.libsvm","r");

        
        
    for(int d =0 ; d < 3; d++){

        setSVMParams(d);

        read_problem(TComDebug::libSVMFile[d]);

	{
         //   param.gamma = 1.0/numFeats[d];
		model[MODE_CU][d] = svm_train(&prob,&param);
		if(svm_save_model(modelPath[MODE_CU][d].c_str(),model[MODE_CU][d]))
		{
			fprintf(stderr, "can't save model to file %s\n", modelPath[MODE_CU][d].c_str());
			exit(1);
		}
		svm_free_and_destroy_model(&model[MODE_CU][d]);
	}


	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);
    }
        
        TComDebug::libSVMFile[0] = fopen("online_d0.libsvm","a");
        TComDebug::libSVMFile[1] = fopen("online_d1.libsvm","a");
        TComDebug::libSVMFile[2] = fopen("online_d2.libsvm","a");

}

void TEncSVM::setSVMParams(int depth){
        
//    if (depth == 0) param.C = 1.0;
//    else if (depth == 1) param.C = 1.0;
//    else param.C = 1.0;
//    
//    if (depth == 0) param.gamma = 1/6.0;
//    else if (depth == 1) param.gamma = 1/6.0;
//    else param.gamma = 1/6.0;
        
    param.C = 1.0;
    param.gamma = 1.0/MAX_ONLINE_FEATURES[depth];	// 1/num_features

    
    
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;



    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 1;

    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
//        param.nr_weight = 2;
//	param.weight_label = Malloc(int,2);
//	param.weight = Malloc(double,2);
//        
//        param.weight_label[0] = 0;
//        param.weight_label[1] = 1;
//        param.weight[0] = 1;
//        param.weight[1] = 5;
        
}
int TEncSVM::readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL){
		
                return 0;
        }
	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
        return 1;
	
}


/* 
 * File:   TEncSVM.h
 * Author: grellert
 *
 * Created on May 5, 2016, 10:32 AM
 */

#ifndef TENCSVM_H
#define	TENCSVM_H

#define ONLINE_SVM 1
#define SVM_TRAINING_SIZE 32
#define SVM_START_PIC 4
#define SVM_REFRESH_GOP (4*SVM_TRAINING_SIZE)

#include "../Lib/TLibEncoder/svm.h"
#include "../Lib/TLibCommon/TComDataCU.h"
#include "../Lib/TLibCommon/TComDebug.h"
#include <string>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <map>
#include <fstream>


#define SVM_PU 0
#define SVM_TU 0
#define SVM_FME 0

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

typedef enum SVM_MODES{
    MODE_CU,
#if SVM_PU
    MODE_PU,
#endif
#if SVM_TU
    MODE_TU,
#endif
#if SVM_FME
    MODE_FME,
#endif
    NUM_SVM_MODES
}SvmMode;

typedef enum TU_MODES{
    TU_D0,
//    TU_D01,
    //TU_D1,
    //TU_D12,
    //TU_D2,
    FULL_TU,
}TuMode;


typedef enum FME_MODES{
    //NO_FME,
    FME_HALF,
   // FME_QUART,
    FULL_FME,
}FmeMode;

const int MAX_ONLINE_FEATURES[] = {3,12,3,3};
const int NUM_LABELS[4] = {2,2,2,2};
const std::string SVM_NAMES[4] = {"CU","PU","TU","FME"};

const int featureOrder_[][33] = {{3, 18, 38, 37, 33, 36, 34, 32, 35, 5, 29, 15, 28, 14, 13, 8, 20, 10, 9, 16, 17, 27, 25, 24, 31, 23, 26, 6, 22, 30, 11, 7, 12},\
                          {3, 18, 37, 38, 33, 36, 34, 5, 32, 15, 29, 35, 14, 13, 8, 28, 10, 9, 16, 17, 27, 12, 7, 24, 26, 25, 22, 31, 11, 23, 30, 6, 20},\
                          {3, 5, 18, 37, 15, 33, 36, 38, 34, 32, 8, 14, 13, 35, 29, 28, 10, 9, 27, 25, 12, 7, 17, 16, 20, 19, 24, 23, 22, 30, 26, 31, 6}};        


class TEncSVM{
public:
    static char *line;
    static int max_line_len;
    static struct svm_parameter param;		// set by parse_command_line
    static struct svm_problem prob;		// set by read_problem
    static struct svm_node *x_space;  

    
    
    static vector<int> featureOrder[3];
    static struct svm_model* model[NUM_SVM_MODES][4];
    static std::string modelPath[NUM_SVM_MODES][4];
    static std::string rangePath[NUM_SVM_MODES][4];
    static std::ostringstream feat_debug_str;

    static bool enableSVM[NUM_SVM_MODES];
    static bool encodeStarted;
    static bool useScaling, structs_allocated;
    static int enabled;
    static int leftSplit;
    static int upSplit;
    static int pictureCount;
    static double splitTh[NUM_SVM_MODES][4];
    static struct svm_node* svmNode;
    static double* probEstimates;
    static std::map<std::string, int> colocSplitMap;
    static int* featIdx[NUM_SVM_MODES][4];
    static double* featMin[NUM_SVM_MODES][4];
    static double* featMax[NUM_SVM_MODES][4];
    static int numFeatures[NUM_SVM_MODES][4];
    static int labelCount[NUM_SVM_MODES][4];
    static int totalCalls[4];


    static void init();
    static bool trainSVM();
    static bool parseRangeModel(std::string path, int mode,int depth);

    static void updateMinMax(TComDataCU *&cu);

    static int predictLabel(TComDataCU *&cu, int mode);
    static void setCUFeatures(TComDataCU *&cu, int mode, int depth);
    static double scale(double value, int index, int mode,int depth);
    static void printSVMNode(int depth);
    static bool isTrainingPic();
    
       
    static void read_problem(FILE *fp);
    static void runSVMTrain();
    static void setSVMParams(int depth);
    static int readline(FILE *input);

};

#endif	/* TENCSVM_H */


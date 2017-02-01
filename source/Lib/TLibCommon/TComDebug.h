/* 
 * File:   TComDebug.h
 * Author: grellert
 *
 * Created on July 19, 2016, 2:25 PM
 */

#ifndef TCOMDEBUG_H
#define	TCOMDEBUG_H
#include <string>
#include <vector>
#include <map>
#include <cstdio>
#include <cmath>
#include "TComDataCU.h"
#include "TComPic.h"
#include "TComYuv.h"

#define EN_DEBUG_215 1

#define DEBUG_POC_START 2

/*const std::string featureMap[] = {"POC","X","Y","W","D","dQP","PRED_MODE","PU_SIZE","BITS","DISTORTION",\
                                "BEST_COST","COST_2Nx2N","COST_MSM","RATIO_2Nx2N-MSM","REL_RATIO_2Nx2N-MSM",\
                                "RATIO_BEST_2Nx2N","RATIO_BEST-MSM", "REL_RATIO_BEST-MSM","TU_DEPTH","TR_SKIP",\
                                "TIME_ms","REF_LIST","REF_IDX","MV_INT_H","MV_INT_V","MV_FRAC_H","MV_FRAC_V",\
                                "MV_MOD","PRED_MV_H","PRED_MV_V","PRED_MV_FRAC_H","PRED_MV_FRAC_V","PRED_MV_MOD",\
                                "MVD_H","MVD_V","MVP_IDX","INTER_DIR", "FME","COLOC_SPLIT","UP_SPLIT", "LEFT_SPLIT",\
                                "UPLEFT_SPLIT","UPRIGHT_SPLIT","CTX_SPLIT", "SPLIT"};
  */
const std::string featureMap[] = {"POC","X","Y","W","D","dQP","PRED_MODE","PU_SIZE","BITS","DISTORTION","BEST_COST",\
                                "SAD","SSE","MSE","COST_2Nx2N","COST_MSM","RATIO_2Nx2N-MSM","REL_RATIO_2Nx2N-MSM",\
                                "RATIO_BEST_2Nx2N","RATIO_BEST-MSM", "REL_RATIO_BEST-MSM","TU_DEPTH","TR_SKIP","NON-Z-COEFF",\
                                "TIME_ms","REF_IDX","MV_MOD_INT","MV_MOD_FRAC","PREDMV_MOD_INT","PREDMV_MOD_FRAC",\
                                "MVD_MOD_INT","MVD_MOD_FRAC","MVP_IDX","INTER_DIR", "FME","COLOC_SPLIT","UP_SPLIT",\
                                "LEFT_SPLIT","UPLEFT_SPLIT","UPRIGHT_SPLIT","CTX_SPLIT", "AVG_NEIGH_DEPTH",\
                                "SVM_PU_UP","SVM_TU_UP","SVM_FME_UP",\
                                "SVM_CU","SVM_PU","SVM_TU","SVM_FME","FINAL_PU","FINAL_TU", "FINAL_FME","SPLIT"};

const int N_FEATURES = sizeof(featureMap) / sizeof(std::string);
const int N_LABELS = 4;

class TComDebug{
public:
    static std::string debugFilePath;
    static FILE* debugFile;
    static FILE* libSVMFile[4];
    static bool encodeStarted, debugSVM;
    static double cost_2Nx2N, cost_MSM;
    static double delta_QP, cu_time, avgNeighborDepth;
    static int numCodedPics;
    static std::map<std::string, std::vector<double> > statsMap; 
    static std::vector<std::string > cuOrderMap; 
    static int splitCU, splitPU, splitTU, splitFME;
    static double SAD, SSE, MSE, nonZeroCoeff;
    static void printHeader();
    static void writeStats();
    static void writeStatsLibSVM();
    static void EVAL_CU(TComDataCU *&cu);
    static double getFeatureValue(TComDataCU *&cu, int feat_idx);
    static double getAverageNeighborDepth(TComDataCU *&cu);
    static double calcAverageDepthIterative(TComDataCU *&cu);
    static void getMSE( TComYuv *pcYuvSrc0, TComYuv *pcYuvSrc1,int width);
    static int getFME(TComDataCU *&cu);

    static std::string getMapString(TComDataCU *cu, int depth, int partIdx);
};

#endif	/* TCOMDEBUG_H */


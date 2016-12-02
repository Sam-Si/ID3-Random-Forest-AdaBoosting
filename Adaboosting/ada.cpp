#include <bits/stdc++.h>

using namespace std;

#define NUM_CLASSIFIER 50 // number of classifiers
#define DROWS 32561 // number of training examples
#define TROWS 16281 //number of testing examples
#define COLS 14

#define re resize
#define hardBoostSize 1

typedef struct TNODE
{
    vector<TNODE*> child;
    vector<int> left;
    int used[20];
    int attrInd;
    int out;
} TNODE;

int hardBoost[hardBoostSize],totalSamples;

double coeffs[NUM_CLASSIFIER],weights[DROWS];
TNODE classifiers[NUM_CLASSIFIER];

int trainData[DROWS][COLS], trainResult[DROWS], samples[DROWS];
int testData[TROWS][COLS], testResult[TROWS];
vector<int> attributeList[COLS];
int cnt_edges;
int samples_training[DROWS];
int samples_current[DROWS][COLS], output_current[DROWS];

// FUNCTION PROTOTYPES
void initializeFunc();
void initializeTree(TNODE *root);
double infoGain(int sample_current[][COLS], int output_current[COLS], vector <int> left , int attr);
int maximumGain(int sample_current[][COLS], int output_current[COLS], vector <int> left , int used[]);
int build_ID3(int sample_current[][COLS], int output_current[COLS], TNODE *cur_node);
int findProb_Accuracy();
int getOutput(int inp[], TNODE* root);
int classifierFunction(int classifier_num);
void boostingFunc();
int getBoostedOutput(int example[]);
void getAccuracy(int examples[][COLS], int output[], int num);

int main () //***done***
{
	//load training examples and their actual output in trainData and trainResult
	freopen("mdata.txt","r",stdin);
	for(int i=0;i<DROWS;i++)
	{
        	for(int j=0;j<=COLS;j++)
        	{
        		if(j==COLS)
			{
                		scanf("%d",&trainResult[i]);
			}
            		else
                		scanf("%d",&trainData[i][j]);
        	}
    	}
	fclose(stdin);
	//load testing examples
	freopen("mtest.txt","r",stdin);
	for(int i=0;i<TROWS;i++)
	{
	        for(int j=0;j<=COLS;j++)
	        {
        		if(j==COLS)
		        {
                		scanf("%d",&testResult[i]);
			}
            		else
                		scanf("%d",&testData[i][j]);
        	}
    	}
	fclose(stdin);
	//adaptive boost function call

	int tempBoost = 4500;
	for (int i = 0; i < hardBoostSize; ++i)
	{
		hardBoost[i] = tempBoost;
		tempBoost += 250;
	}

	for(int i=0;i<hardBoostSize; i++)
	{
		//cout << i+1 << ":" <<endl;
		totalSamples = hardBoost[i];
	
		boostingFunc();
	
		getAccuracy(trainData, trainResult, DROWS);

		getAccuracy(testData, testResult, TROWS);

	}

return 0;
}





double infoGain(int sample_current[][COLS], int output_current[COLS], vector <int> left , int attr){
    	double info_gain = 0;
    	int posCnt = 0 , negCnt = 0;
    	for(int i = 0 ; i < left.size() ; i++)
		{
        	int inp_ind = left[i];
        	if(output_current[inp_ind]==1)posCnt++;
        	else negCnt++;
    	}
    	double p_pos = (posCnt*1.0)/(left.size()*1.0);
    	double p_neg = (negCnt*1.0)/(left.size()*1.0);
    	double initial_entropy = -1.0 * ( (p_pos*log(p_pos)) + (p_neg*log(p_neg)) );
    	if(posCnt == 0 || negCnt == 0)
		{
        	initial_entropy = 0; //  pure subset , hence leaf TNODE
    	}
    	// now calculate the weighted average entropy after split
    	double weighted_av = 0;
    	for(int i = 0 ; i < attributeList[attr].size() ; i++ )
		{
        	posCnt = 0;
        	negCnt = 0;
        	for(int j = 0 ; j < left.size() ; j++)
			{
        		int inp_ind = left[j];
            		if(sample_current[inp_ind][attr]==attributeList[attr][i])
					{
	                	if(output_current[inp_ind]==1)posCnt++;
	                	else negCnt++;
	            	}
	        }
	        int tot = posCnt + negCnt ;
	        p_pos = (posCnt*1.0)/(tot*1.0);
	        p_neg = (negCnt*1.0)/(tot*1.0);
	        double child_attr_entropy = -1.0 * ( (p_pos*log(p_pos)) + (p_neg*log(p_neg)) );
	        if(negCnt == 0 || posCnt == 0)child_attr_entropy = 0;
	        weighted_av+= (-1.0) * ((tot*1.0)/(left.size()*1.0)*child_attr_entropy);
    	}

    	//cout << initial_entropy << " " << weighted_av <<  endl;
    	info_gain = initial_entropy + weighted_av ;
    	//cout << info_gain << endl;
    	return info_gain;
}

int maximumGain(int sample_current[][COLS], int output_current[COLS], vector <int> left , int used[]){
    	double max_gain = 0;
    	int attrInd = -1;
    	for(int i = 0 ; i < COLS ; i++)
	{
        	if(used[i])continue;
        	else
		{
            		double gain = infoGain(sample_current, output_current, left,i);
            		if(gain==1.0)return i;
            		if(gain > max_gain)
			{
                		max_gain = gain;
                		attrInd = i;
            		}
        	}
    	}
   	return attrInd;
}

void initializeTree(TNODE *root)
{
    	(root->left).re(totalSamples);
	root->out=-1;
	for(int i=0;i<14;i++) root->used[i]=0;
    	for(int i=0;i<totalSamples;i++) root->left[i]=i;
}


int findProb_Accuracy()
{
	double cumWeights[DROWS] = {0};
	int index = DROWS-1;
	cumWeights[0] = weights[0];
	for(int i=1; i<DROWS; i++)
	{
		cumWeights[i] = cumWeights[i-1]+weights[i];
	}
	double random = (rand()%100)/100.0;
	while(index>=0)
	{
		if(cumWeights[index]<random)
			break;
		index--;
	}
	return (index+1)%DROWS;
}

void boostingFunc()
{
	srand(time(NULL));
	for(int i=0; i<DROWS; i++)
		weights[i] = 1/((double)DROWS);
	initializeFunc();
	//cout << 1.0/DROWS << endl;
	for(int i=0; i<NUM_CLASSIFIER; i++)
	{
	//	cout << i << ":" <<endl;
		if(classifierFunction(i))i--;
	}
}

//to get the output for an example for a given classifier
int getOutput(int inp[], TNODE* root)
{
	int val=root->attrInd;
	if(val==-1)
	{
	        return root->out;
	}

	return getOutput(inp, root->child[inp[val]]);
}


int classifierFunction(int classifier_num) 
{
	//srand(time(NULL));
	int count_neg = 0;
	double error = 0, weights_sum = 0;
	//return ;
	for(int i=0; i<totalSamples; i++)
	{
		samples_training[i] = findProb_Accuracy();
	}
	for(int i=0; i<totalSamples; i++)
	{
		for(int j=0; j<COLS; j++)
			samples_current[i][j] = trainData[samples_training[i]][j];
		output_current[i] = trainResult[samples_training[i]];
	}
	TNODE classifier;
	initializeTree(&classifier);
	cnt_edges = 0;
	build_ID3(samples_current, output_current, &classifier);
	classifiers[classifier_num] = classifier;
	for(int i=0; i<DROWS; i++)
	{
		double temp = getOutput(trainData[i],&classifiers[classifier_num]);
		if(temp!=trainResult[i])
		{
			count_neg++;
			error+= weights[i];
		}
		weights_sum+=weights[i];
	}
	//cout << count_neg << endl;
	//cout << error << " " << weights_sum << endl;
	coeffs[classifier_num] = log((1.0-error)/error)/(2.0);
	//update weights
	double sum = 0;
	//cout << coeffs[classifier_num] << endl;
	for(int i=0; i<DROWS; i++)
	{
		double temp = getOutput(trainData[i],&classifiers[classifier_num]);
		if(temp==0) temp=-1;
		weights[i] = weights[i]*exp(-(trainResult[i]==0?-1:1)*coeffs[classifier_num]*temp);
		
		sum+=weights[i];
	}
	//normalize weights
	for(int i=0; i<DROWS; i++)
	{
		weights[i]/=sum;
	}
	if(coeffs[classifier_num]<=0)
	{
		return 1;
	}
	return 0;
}

int build_ID3(int sample_current[][COLS], int output_current[COLS], TNODE *cur_node){
    	vector <int> left(cur_node->left);
    	cnt_edges++;
    	int max_gain_attr = maximumGain(sample_current, output_current, left,cur_node->used);
    	if(max_gain_attr == -1)
	{ // no more possible attributes to split
        	int count_one = 0 , count_zero = 0;
        	for(int i = 0 ; i < left.size() ; i++)
		{
            		int inp_ind = left[i];
            		if(output_current[inp_ind]==1)count_one++;
            		else count_zero++;
        	}
        	cur_node->attrInd = -1;
        	if(2*count_zero >= left.size())cur_node->out = 0;
        	else cur_node->out = 1;
        	return 0;
    	}
    	cur_node->attrInd = max_gain_attr;
    	cur_node->used[max_gain_attr] = 1;
    	for(int i = 0 ; i < attributeList[max_gain_attr].size() ; i++)
	{
        	TNODE *temp = new TNODE;
        	temp->child.clear();
        	temp->left.clear();
        	memset(temp->used,0,sizeof(temp->used));
        	for(int j = 0 ; j < COLS ; j++)
		{
            		temp->used[j] = cur_node->used[j];
        	}
        	temp->out = -1;
        	for(int j =  0 ; j < left.size() ; j++)
		{
            		int inp_ind = left[j];
            		if(sample_current[inp_ind][max_gain_attr] == attributeList[max_gain_attr][i])
			{
                		temp->left.push_back(inp_ind);
            		}
        	}
        	cur_node->child.push_back(temp);
        	build_ID3(sample_current, output_current, temp);
    	}
    	return 0;
}


//Adaptive Boosting function


int getBoostedOutput(int example[])
{
	double outputs[NUM_CLASSIFIER] = {0};
	for(int i=0; i<NUM_CLASSIFIER; i++)
	{
		outputs[i] = getOutput(example, &classifiers[i]);
		if(outputs[i]==0) outputs[i] = -1;
	}
	double output = 0;
	for(int i=0; i<NUM_CLASSIFIER; i++)
	{
		output+=outputs[i]*coeffs[i];
	}
	if(output<0)
		return 0;
	return 1;
}

//To get the accuracy of the boosted ensembler on a set of examples
void getAccuracy(int examples[][COLS], int output[], int num)
{
	int count = 0;
	for(int i=0; i<num; i++)
	{
		int temp = getBoostedOutput(examples[i]);
		if(temp==output[i])
			count++;
	}
	cout << "Accuracy: " << (count*100.0)/num << '%' << endl;
}


void initializeFunc()
{
	attributeList[0].re(4); // continous
	attributeList[1].re(8);
	attributeList[2].re(2); // continous
	attributeList[3].re(16);
	attributeList[4].re(2); // continous
	attributeList[5].re(7);
	attributeList[6].re(14);
	attributeList[7].re(6);
	attributeList[8].re(5);
	attributeList[9].re(2);
	attributeList[10].re(2); // continous
	attributeList[11].re(2); // continous
	attributeList[12].re(2); // continous
	attributeList[13].re(41);
	for(int i=0;i<COLS;i++)
	        for(int j=0;j<attributeList[i].size();j++)
        		attributeList[i][j]=j;
}
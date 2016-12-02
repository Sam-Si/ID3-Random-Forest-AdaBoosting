#include <bits/stdc++.h>

using namespace std;

// ********************************************************************** //

// Tree NODE

struct TNODE{
	vector<int> data_index; //Indices of values taken by the particular TNODE
	int divide_f;
    bool att_present[15]; // Indicates which of the attributes have already been assumed
    map<string, TNODE*> children;
    int target_v;
    int current_max;
    TNODE(){
    	current_max=-1;
        target_v=-1; 
        divide_f=-1;
        memset(att_present, false, sizeof(att_present)); // Array Initialization
        
    }
};

// ***************************************************************************//

// DataTable

typedef vector<string> VS;
typedef vector<VS> VVS;
typedef vector<int> VI;
typedef vector<VI> VVI;

// Class to store training

class DataTable{
	public:
		 int DROWS;			// No. of rows in the training data
		 VVS data; 					//	(DROWS+1)x(14)...Row[0] contains names of attributes. Range from 1 to DROWS. Cols from 0-14
		 VI result;					//	(DROWS+1)
		 long long thresholdVals[15];		// thresholdVals for continuous values
	public:
	DataTable(int rec){
		DROWS=rec;
		data.resize(DROWS+1,vector<string> (15)); 
		result.resize(DROWS+1);
		memset(thresholdVals,0,sizeof(thresholdVals));
	}

	// Class Functions...
	void set(int idx,int attribute,string attrVal);
    void printData();
    void printResult();
    void initFeature();
	void modify_column(int);
	void replaceMissing();
	void modifyTestdata(vector<string> &);
};

void DataTable::set(int idx,int attribute,string attrVal)
{
	if(attribute==15)
	{
		if (attrVal==" >50K")
			DataTable::result[idx]=1;
		else
			DataTable::result[idx]=0;
	}
	else
		DataTable::data[idx][attribute]=attrVal;
}

void DataTable::printData()
{
    for(int i=1;i<=DROWS;i++)
    {
        for(int j=1;j<=14;j++)
            cout<<DataTable::data[i][j]<<"/";
        cout<<endl;
    }
}
void DataTable::printResult()
{
    for(int i=1;i<=DROWS;i++)
    {
        cout<<DataTable::result[i]<<"\n";
    }
}

void DataTable::replaceMissing(){
	
	for(int col=1;col<15;col++){
		map<string,int> count;
		for(int i=1;i<=DROWS;i++){
			if(count.find(data[i][col])==count.end())
				count[data[i][col]]=1;
			else
				count[data[i][col]]++;
		}
		string mostFreq;
		int maxfreq=0;
		map<string,int>::iterator it;
		for(it=count.begin();it!=count.end();it++){
			if(it->second > maxfreq){
				maxfreq = it->second;
				mostFreq = it->first;
			}
		}
		for(int i=1;i<=DROWS;i++){
			if(data[i][col]==" ?")
				data[i][col]=mostFreq;
		}
	}
}

void DataTable::modify_column(int col){
	long long total=0;
	for(int i=1;i<=DROWS;i++){
		total += (long long)stoi(data[i][col].substr(1));
	}
	long long threshold = (total/DROWS);	
	thresholdVals[col]=threshold;
	for(int i=1;i<=DROWS;i++){
		if((long long)stoi(data[i][col].substr(1)) > threshold){
			data[i][col]=" 1";		// Class1
		}
		else{
			data[i][col]=" 0";		//class0
		}
	}
}

// CONTINUOUS DATA

void DataTable::modifyTestdata(vector<string> &values){
	int cont[]={1,3,5,11,12,13};
	for(int i=0;i<6;i++){
		if(cont[i]!=1)	
			values[cont[i]-1]= values[cont[i]-1].substr(1);	// remove space
		values[cont[i]-1] = " "+to_string(  (long long) stoi(values[cont[i]-1]) > thresholdVals[cont[i]]);
	}
}

// ***************************************************************************//

// TREE

// Function Prototypes involved in building of the tree
double getEntropyElmnt(int idx,TNODE &current,DataTable &t);
void initDivide_f(TNODE &current,DataTable &t);
TNODE* getNode(bool present[],vector<int> &Examples);
TNODE* buildTree(vector<int> &Examples,DataTable &t,bool tr[]);
TNODE* treeConst(DataTable &t,bool);
void printTree(TNODE *root);
void printNode(TNODE *n);
int getNextNode(TNODE *,vector<string>);
double getAccuracy(vector<int>,vector<int>);

double getEntropyElmnt(int att_number,TNODE &current,DataTable &t)
{
    unordered_map<string,int> pos_neg[2];
    unordered_map<string,int> counts;
    double S=0;
	for(int i=0;i<current.data_index.size();i++)
	{
        S++;
		counts[t.data[current.data_index[i]][att_number]]++;
        if (t.result[current.data_index[i]]==0) {
            pos_neg[0][t.data[current.data_index[i]][att_number]]++;
        }
        else
        {
            pos_neg[1][t.data[current.data_index[i]][att_number]]++;
        }
	}
	double entropy=0;
	
	for(unordered_map<string, int>::iterator it=counts.begin();it!=counts.end();it++)
	{
		double Sv=(it->second); //Sv
        double denom=pos_neg[0][it->first]+pos_neg[1][it->first];
        double pos=pos_neg[1][it->first];
        double neg=pos_neg[0][it->first];
        if(pos==0)pos=1;
        if(neg==0)neg=1;
        entropy+=(Sv/S)*(-(pos/denom)*(log(pos/denom)/log(2))-(neg/denom)*(log(neg/denom)/log(2)));
	}
	return entropy;
}

void initDivide_f(TNODE &current,DataTable &t)
{
	double min_entropy=numeric_limits<double>::max();
	for(int i=1;i<15;i++)
	{
		if(current.att_present[i])
		{
			double temp=getEntropyElmnt(i,current,t);
            /*if(temp!=temp)
            {
                current.divide_f=i;
                break;
            }*/
			if(min_entropy>temp)
			{
				min_entropy=temp;
				current.divide_f=i;
			}
		}
	}
}

TNODE* getNode(bool present[],vector<int> &Examples)
{
    TNODE *temp=new TNODE;
	for(int i=0;i<15;i++)
	temp->att_present[i]=present[i];
    temp->data_index=Examples;
	return temp;
}

TNODE* buildTree(vector<int> &Examples,DataTable &t,bool attributes[])
{

	int s=0;
	bool flag=false;
	//initDivide_f(current,t);
    TNODE *current=getNode(attributes,Examples);
    for(int i=1;i<15;i++)
        flag|=attributes[i];
	for(int i=0;i<Examples.size();i++)
	{
		s+=t.result[Examples[i]];
		
	}
	if(!flag || s==0 || s==Examples.size())
	{
		current->target_v=(s>(Examples.size()/2.0));
		return current;
	}
	current->current_max=(s>(Examples.size()/2.0));
    initDivide_f(*current, t);
    current->att_present[current->divide_f]=false;
    unordered_map<string, vector<int> > subsets;
	for(int i=0;i<Examples.size();i++)
	{
		subsets[t.data[Examples[i]][current->divide_f]].push_back(Examples[i]); //val of att
	}
    
    for(unordered_map<string, vector<int> >::iterator it=subsets.begin();it!=subsets.end();it++)
	{
        current->children[it->first]=buildTree(it->second, t, current->att_present);   
	}
    return current;
}

TNODE* treeConst(DataTable &t, bool random)
{
	vector<int> dataset;
	for(int i=1;i<=t.DROWS;i++)
		dataset.push_back(i);
	bool all_true[15];
	if(random){						// If Random forest, select few, else select all
		srand(time(NULL));
		memset(all_true,false,sizeof all_true);
		for(int i=0;i<4;i++)						// NUMBER OF FEATURES FOR RANDOM
			all_true[1+rand()%14]=true;
	}
	else
		memset(all_true,true,sizeof all_true);
    //all_true[3]=false;all_true[1]=false;all_true[13]=false;all_true[11]=false;all_true[5]=all_true[12]=false;
	TNODE* root=buildTree(dataset,t,all_true);
	return root;
}

void printNode(TNODE *toPrint)
{
    cout<<"Result "<<toPrint->target_v<<endl;
    cout<<"Dividing atrribute "<<toPrint->divide_f<<endl;
    
}

void printTree(TNODE *current)
{
    if (current==nullptr) {
        return;
    }
    printNode(current);
    for (map<string,TNODE*>::iterator it=current->children.begin();it!=current->children.end();it++) {
        printTree(it->second);
    }
}

int getNextNode(TNODE *root, vector<string> values){
			
    if(root->target_v != -1 || root->divide_f==-1)		// if leaf TNODE
    	return root->target_v;
    
    string splitting_col_value = values[root->divide_f - 1];			// values is 0 indexed
    if(root->children[splitting_col_value]==NULL)
    	return root->current_max;
	return getNextNode(root->children[splitting_col_value],values);
}

double getAccuracy(vector<int> actual,vector<int> resultArray){
	if(actual.size()!=resultArray.size()){
		cout<<" \nDIFFERENT SIZES OF RESULTS\n";
		return -1.0;
	}
	int cnt=0;
	for(int i=0;i<actual.size();i++){
		cnt += (resultArray[i]==actual[i]);
	}
	return ((double)(cnt)*100)/actual.size();
}

// ***************************************************************************//

// CONSTANTS AND GLOBAL VARIABLES

const int num_Trees=40;				// Number of decision trees
const int TSIZE = 32561;		//  SIZE of input file

const int FTSIZE=10000;         // Forest DataTable Size
const double THRESHOLD=0;

DataTable dataTable(TSIZE);			// Main DataTable which contains all data

// ***************************************************************************//

void parser(string line,int id){
    stringstream l(line);
    string attrVal;
    char delim=',';
    int att_number=1;
    
    while(getline(l,attrVal,delim))
    {
        dataTable.set(id,att_number,attrVal);
        att_number++;
    }
}

vector<string> getTokenized(string line){
	vector<string> ans;
	stringstream l(line);
    string attrVal;
    char delim=',';
	while(getline(l,attrVal,delim)){
		ans.push_back(attrVal);
	}
	return ans;
}

void processDataset()
{
    dataTable.replaceMissing();    
    int cont[]={1,3,5,11,12,13};
    for(int i=0;i<6;i++){
        dataTable.modify_column(cont[i]);
    }
}
double weight_f(double accuracy)
{
    if(accuracy==1)
        return 1;
    else
        return 0.5*log(accuracy);
}


int main(int argc, const char * argv[])
{

    string line;
    string filename="data.txt";
    ifstream dataFile(filename);
    
    int idx=1;
    if(dataFile.is_open())
    {
        while(getline(dataFile,line)){
            //parse contents
			parser(line,idx);
			idx++;
        }
        dataFile.close();
    }
    else
    {
        cout<<"Problem opening File!\n";
        return -1;
    }
    cout<<"Initialising.....\n";
    
	processDataset();
	//dataTable.printData();
	
	srand(time(NULL));
  	
  	DataTable *t[num_Trees];	
  	
  	for(int i=0;i<num_Trees;i++){
	  	t[i]=new DataTable(FTSIZE);						// Number of examples to be considered for every decision tree 	
	  	for(int k=1;k <= (t[i]->DROWS) ; k++){
			int row = 1+(rand()%TSIZE);
			t[i]->data[k] = dataTable.data[row];
			t[i]->result[k] =dataTable.result[row];
		}
	}
    
    TNODE *root[num_Trees];						// Array of Root Nodes
    for(int i=0;i<num_Trees;i++){
		root[i] = treeConst(*t[i],false);    	// false for boost
    }
    
    /***Generates weight vector**********/

    
    vector<double> weights;                     
    for(int i=0;i<num_Trees;i++)
    {
        double N=t[i]->DROWS;  
        double accuracy=0; 
        for(int k=1;k<=(t[i]->DROWS);k++)
        {
            vector<string> current_values;
            for(int j=1;j<=14;j++)
            current_values.push_back(t[i]->data[k][j]);
            int p=getNextNode(root[i],current_values);
            if(p==t[i]->result[k])
            {
                accuracy++;
            }
        }
        if(N!=0)
        accuracy/=N;
        weights.push_back(weight_f(accuracy));
    }


    /************************************/
	// TESTING DATA

	ifstream testFile("test.txt");
	idx=1;
	
	vector< vector<int> > resultArray;		// resultArray[i][j]= ith example , jth tree
	vector<int> actual;						// actual[i] = ith output
	if(testFile.is_open())
    {

        while(getline(testFile,line)){
        
            vector<string> values = getTokenized(line);
            actual.push_back(values.back() == " >50K.");
            values.pop_back();					
            vector<int> temp;					
            for(int i=0;i<num_Trees;i++){
	            t[i]->modifyTestdata(values);	
				int p = getNextNode(root[i],values);
				temp.push_back(p);
			}
			resultArray.push_back(temp);
        }
        testFile.close();
    }
    else
    {
        cout<<"Problem opening testing File!\n";
        return -1;
    }

    // CALCULATING THE RESULT FOR THE BOOSTED FOREST
    vector<int> boosted;
    vector<double> temp(num_Trees,0);
    for(int i=0;i<resultArray.size();i++)
    {
        temp.push_back(0);
        for(int j=0;j<resultArray[i].size();j++)
        {
            temp[i]+=weights[j]*((resultArray[i][j])?(1):(-1));
        }
        if(temp[i]>THRESHOLD)
            boosted.push_back(0);
        else if(temp[i]<-THRESHOLD)
            boosted.push_back(1);
        else boosted.push_back(1);
    }

    //*******************************PROG END***********************************//
    
	double acc = getAccuracy(actual,boosted);
	cout<<"Accuracy of Random Forest: " << acc << endl;

    return 0;
}

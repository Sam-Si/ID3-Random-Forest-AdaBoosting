#include <bits/stdc++.h>

using namespace std;

// ************************************************************************//
// fork

struct fork{
	vector<int> data_index; //index of data items pertaining to this fork
	int divide_f;
    bool attribute_current[15];
    map<string, fork*> children;
    int target_v;
    int current_max;
    fork(){
    	current_max=-1;
        target_v=-1; //Not a leaf fork hence not 0 or 1
        divide_f=-1;
        memset(attribute_current, false, sizeof(attribute_current));
        
    }
};

// *************************************************************************//
// TABLE

typedef vector<string> VS;
typedef vector<VS> VVS;
typedef vector<int> VI;
typedef vector<VI> VVI;

class Table{
	public:
		 int RECORDS;			// Number of training records
		 VVS data; 					//	(RECORDS+1)x(14)  ***	First row0 contains column names *** row=[1,RECORDS] , column=[0,13]
		 VI result;					//	(RECORDS+1)
		 long long average[15];		// average for continuos values
	public:
	Table(int rec){
		RECORDS=rec;
		data.resize(RECORDS+1,vector<string> (15)); 
		result.resize(RECORDS+1);
		memset(average,0,sizeof(average));
	}
	void initFeature();
	void initz(int idx,int attribute,string att_value);
    void coutData();
    void coutResult();
    
	void change_c(int);
	void UnknownReplacer();
	void TestDataModi(vector<string> &);
};


void Table::initz(int idx,int attribute,string att_value)
{
	if(attribute==15)
	{
		if (att_value==" >50K")
			Table::result[idx]=1;
		else
			Table::result[idx]=0;
	}
	else
		Table::data[idx][attribute]=att_value;
}

void Table::coutData()
{
    for(int i=1;i<=RECORDS;i++)
    {
        for(int j=1;j<=14;j++)
            cout<<Table::data[i][j]<<"/";
        cout<<endl;
    }
}
void Table::coutResult()
{
    for(int i=1;i<=RECORDS;i++)
    {
        cout<<Table::result[i]<<"\n";
    }
}

// NEW FUNCTIONS

void Table::UnknownReplacer(){
	
	for(int col=1;col<15;col++){
		map<string,int> count;
		for(int i=1;i<=RECORDS;i++){
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
		for(int i=1;i<=RECORDS;i++){
			if(data[i][col]==" ?")
				data[i][col]=mostFreq;
		}
	}
}

void Table::change_c(int col){
	long long total=0;
	for(int i=1;i<=RECORDS;i++){
		total += (long long)stoi(data[i][col].substr(1));
	}
	long long threshold = (total/RECORDS);	
	average[col]=threshold;
	for(int i=1;i<=RECORDS;i++){
		if((long long)stoi(data[i][col].substr(1)) > threshold){
			data[i][col]=" 1";		// Class1
		}
		else{
			data[i][col]=" 0";		//class0
		}
	}
}

// CONTINUOUS TEST DATA

void Table::TestDataModi(vector<string> &values){
	int cont[]={1,3,5,11,12,13};
	for(int i=0;i<6;i++){
		if(cont[i]!=1)	
			values[cont[i]-1]= values[cont[i]-1].substr(1);	// remove space
		values[cont[i]-1] = " "+to_string(  (long long) stoi(values[cont[i]-1]) > average[cont[i]]);
	}
}



// *************************************************************************//
// Tree

double FaPEntropy(int idx,fork &current,Table &t);
void initByDivide_ff(fork &current,Table &t);
fork* getNode(bool present[],vector<int> &Examples);
fork* buildTree(vector<int> &Examples,Table &t,bool tr[]);
fork* Construction(Table &t,bool);
void printTree(fork *root);
void printNode(fork *n);
int ClassPredict(fork *,vector<string>);
double FaPAccu(vector<int>,vector<int>);


double FaPEntropy(int att_number,fork &current,Table &t)
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
void initByDivide_ff(fork &current,Table &t)
{
	double min_entropy=numeric_limits<double>::max();
	for(int i=1;i<15;i++)
	{
		if(current.attribute_current[i])
		{
			double temp=FaPEntropy(i,current,t);
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
fork* getNode(bool present[],vector<int> &Examples)
{
    fork *temp=new fork;
	for(int i=0;i<15;i++)
	temp->attribute_current[i]=present[i];
    temp->data_index=Examples;
	return temp;
}
fork* buildTree(vector<int> &Examples,Table &t,bool attributes[])
{

	int s=0;
	bool flag=false;
	//initByDivide_ff(current,t);
    fork *current=getNode(attributes,Examples);
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
    initByDivide_ff(*current, t);
    current->attribute_current[current->divide_f]=false;
    unordered_map<string, vector<int> > subsets;
	for(int i=0;i<Examples.size();i++)
	{
		subsets[t.data[Examples[i]][current->divide_f]].push_back(Examples[i]); //val of att
	}
    
    for(unordered_map<string, vector<int> >::iterator it=subsets.begin();it!=subsets.end();it++)
	{
        current->children[it->first]=buildTree(it->second, t, current->attribute_current);   
	}
    return current;
}
fork* Construction(Table &t, bool random)
{
	vector<int> dataset;
	for(int i=1;i<=t.RECORDS;i++)
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
	fork* root=buildTree(dataset,t,all_true);
	return root;
}
void printNode(fork *toPrint)
{
    cout<<"Result "<<toPrint->target_v<<endl;
    cout<<"Dividing atrribute "<<toPrint->divide_f<<endl;
    
}
void printTree(fork *current)
{
    if (current==nullptr) {
        return;
    }
    printNode(current);
    for (map<string,fork*>::iterator it=current->children.begin();it!=current->children.end();it++) {
        printTree(it->second);
    }
}
int ClassPredict(fork *root, vector<string> values){
			
    if(root->target_v != -1 || root->divide_f==-1)		// if leaf fork
    	return root->target_v;
    
    string splitting_col_value = values[root->divide_f - 1];			// values is 0 indexed
    if(root->children[splitting_col_value]==NULL)
    	return root->current_max;
	return ClassPredict(root->children[splitting_col_value],values);
}

double FaPAccu(vector<int> actual,vector<int> predicted){
	if(actual.size()!=predicted.size()){
		cout<<" \nDIFFERENT SIZES OF RESULTS\n";
		return -1.0;
	}
	int cnt=0;
	for(int i=0;i<actual.size();i++){
		cnt += (predicted[i]==actual[i]);
	}
	return ((double)(cnt)*100)/actual.size();
}


// *************************************************************************//

// MAIN

const int num_d=100;				// Number of decision trees
const int TSIZE = 32561;		//  SIZE of input file
Table maintable(TSIZE);			// Main table which contains all data
const int FTSIZE=10000;         // Forest table Size


void parser(string line,int id){
    stringstream l(line);
    string att_value;
    char delim=',';
    int att_number=1;
    
    while(getline(l,att_value,delim))
    {
        maintable.initz(id,att_number,att_value);
        att_number++;
    }
}

vector<string> extractCells(string line){
	vector<string> ans;
	stringstream l(line);
    string att_value;
    char delim=',';
	while(getline(l,att_value,delim)){
		ans.push_back(att_value);
	}
	return ans;
}


int main(int argc, const char * argv[])
{

    string line;
    string filename="data.txt";
    ifstream file(filename);
    
    int idx=1;
    if(file.is_open())
    {
        while(getline(file,line)){
            //parse contents
			parser(line,idx);
			idx++;
        }
        file.close();
    }
    else
    {
        cout<<"Problem opening File!\n";
        return -1;
    }
    cout<<"Dataset Initialized Successfully\n";
    
	maintable.UnknownReplacer();    
    int cont[]={1,3,5,11,12,13};
    for(int i=0;i<6;i++){
    	maintable.change_c(cont[i]);
	}
	//maintable.coutData();
	
	srand(time(NULL));
  	
  	Table *t[num_d];				// tables
  	
  	for(int i=0;i<num_d;i++){
	  	t[i]=new Table(FTSIZE);						// 10000 examples per decision tree 	
	  	for(int k=1;k <= (t[i]->RECORDS) ; k++){
			int row = 1+(rand()%TSIZE);
			t[i]->data[k] = maintable.data[row];
			t[i]->result[k] =maintable.result[row];
		}
	}
    
    fork *root[num_d];						// Roots
    for(int i=0;i<num_d;i++){
		root[i] = Construction(*t[i],true);    	// true for random forest
    }
    
	// TESTING DATA

	ifstream file2("test.txt");
	idx=1;
	
	vector< vector<int> > predicted;		// predicted[i][j]= ith example , jth tree
	vector<int> actual;						// actual[i] = ith output
	if(file2.is_open())
    {

        while(getline(file2,line)){
        
            vector<string> values = extractCells(line);
            actual.push_back(values.back() == " >50K.");
            values.pop_back();					// remove class label before testing
            vector<int> temp;					// stores predicted values for one row
            for(int i=0;i<num_d;i++){
	            t[i]->TestDataModi(values);	// modifies continuous test data acc to Table t
				int p = ClassPredict(root[i],values);
				temp.push_back(p);
			}
			predicted.push_back(temp);
        }
        file2.close();
    }
    else
    {
        cout<<"Problem opening testing File!\n";
        return -1;
    }
    vector<int> mode;						// mode of predicted values
    for(int i=0;i<predicted.size();i++){
    	int count[2]={};					// counts of 0 and 1
    	for(int j=0;j<num_d;j++){
    		count[predicted[i][j]]++;	
    	}
    	if(count[0]>=count[1])
    		mode.push_back(0);
    	else
    		mode.push_back(1);
    }
    
	double acc = FaPAccu(actual,mode);
	cout << "ID3 accuracy is : " <<acc<< '%' <<endl;

    return 0;
}

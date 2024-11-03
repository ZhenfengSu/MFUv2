#include<iostream>
#include<fstream>
using namespace std;

void test()
{
    ofstream ofs;  
    ofs.open("hhh007.txt",ios::out );
    for ( int i=1; i<264347; i++) {    //加了个for循环去将第一列写成全部Q，第二列写成123...
    ofs << "Q" << "\t" << i << "\t" << endl;  //endl用于换行
	} 

    ofs.close();
}
int main()
{
    test();		//test的东西可以直接写到main函数也可以
    getchar();  //可以不用
    return 0;
}

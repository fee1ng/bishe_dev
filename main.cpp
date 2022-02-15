//#include "D:\Python\Python37\include\Python.h"
//当然，绝对路径永远不会失效！^o^
#include "/usr/include/python3.6m/Python.h"
#include <iostream>
using namespace std;
void cython1()
{
    Py_Initialize();                          //初始化python解释器，告诉编译器要用的python编译器
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");
    PyRun_SimpleString("import hello");       //调用python文件
    PyRun_SimpleString("hello.__name__()"); //调用上述文件中的函数
    Py_Finalize();                            //结束python解释器，释放资源
}

void cython2()
{
    Py_Initialize();
    PyObject* pModule = NULL;
    PyObject* pFunc = NULL;
    PyObject* pArg = NULL;
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");
    pModule = PyImport_ImportModule("hello");              //这里是要调用的文件名
    cout<<"========================="<<endl;
    pFunc = PyObject_GetAttrString(pModule, "__name__()"); //这里是要调用的函数名
    
    pArg = Py_BuildValue("(i)",10);  // 变量格式转换成python格式
    PyEval_CallObject(pFunc,NULL);                        //调用函数
    Py_Finalize();
}

int main()
{
    int select;
    cin >> select;
    select == 1 ? cython1() : cython2();
    return 0;
}

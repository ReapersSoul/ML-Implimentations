#include "TensorLib.hpp"

int main()
{
    Tensor t1({5, 5, 5});
    t1.MakeIndexTensor();
    Tensor t2=t1.Block({0, 0, 0}, {2, 2, 2});
    t1.Print();
    t2.Print();

	return 0;
}
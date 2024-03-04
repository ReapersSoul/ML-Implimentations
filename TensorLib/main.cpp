#include "TensorLib.hpp"

int main()
{
	Tensor t({2, 2});
	t.Randomize(1, 20);
	std::vector<int> s = {1, 1};
	Tensor t2 = t.Block(s, {1, 1});

	t.Print();
	t2.Print();

	return 0;
}
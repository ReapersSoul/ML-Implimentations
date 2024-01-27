#pragma once
#include <random>

//rand range with a proper distribution
static std::random_device rd;
static std::mt19937 gen(rd());
static double RandRange(double min, double max) {
	std::uniform_real_distribution<> dis(min, max);
	return dis(gen);
}
#include <iostream>
#include <thread>
#include <chrono>

#include "RecurrentLayer.hpp"
#include "../0-Helpers/LossFunction.hpp"

// sdl2
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

class Sigmoid : public ActivationFunction
{
public:
	double Activate(double input)
	{
		return 1.0 / (1.0 + exp(-input));
	}
	double Derivative(double input)
	{
		return Activate(input) * (1.0 - Activate(input));
	}

	std::vector<double> Activate(std::vector<double> input)
	{
		std::vector<double> output(input.size());
		for (int i = 0; i < input.size(); i++)
		{
			output[i] = Activate(input[i]);
		}
		return output;
	}
	std::vector<double> Derivative(std::vector<double> input)
	{
		std::vector<double> output(input.size());
		for (int i = 0; i < input.size(); i++)
		{
			output[i] = Derivative(input[i]);
		}
		return output;
	}
};

class MSE : public LossFunction
{
public:
	double Calculate(double output, double target)
	{
		return pow(output - target, 2);
	}
	double Derivative(double output, double target)
	{
		return 2 * (output - target);
	}

	std::vector<double> Calculate(std::vector<double> output, std::vector<double> target)
	{
		std::vector<double> loss(output.size());
		for (int i = 0; i < output.size(); i++)
		{
			loss[i] = Calculate(output[i], target[i]);
		}
		return loss;
	}
	std::vector<double> Derivative(std::vector<double> output, std::vector<double> target)
	{
		std::vector<double> loss(output.size());
		for (int i = 0; i < output.size(); i++)
		{
			loss[i] = Derivative(output[i], target[i]);
		}
		return loss;
	}
};

#define Zero 0.00000001
#define One 1

struct Color
{
	double r, g, b;

	static Color HSV(double h, double s, double v)
	{
		double c = v * s;
		double x = c * (1 - abs(fmod(h / 60.0, 2) - 1));
		double m = v - c;
		double rp, gp, bp;
		if (h >= 0 && h < 60)
		{
			rp = c;
			gp = x;
			bp = 0;
		}
		else if (h >= 60 && h < 120)
		{
			rp = x;
			gp = c;
			bp = 0;
		}
		else if (h >= 120 && h < 180)
		{
			rp = 0;
			gp = c;
			bp = x;
		}
		else if (h >= 180 && h < 240)
		{
			rp = 0;
			gp = x;
			bp = c;
		}
		else if (h >= 240 && h < 300)
		{
			rp = x;
			gp = 0;
			bp = c;
		}
		else
		{
			rp = c;
			gp = 0;
			bp = x;
		}
		return Color(rp + m, gp + m, bp + m);
	}

	Color(double r, double g, double b)
	{
		this->r = r;
		this->g = g;
		this->b = b;
	}
	Color(double h, double s, double v, bool hsv)
	{
		Color c = HSV(h, s, v);
		this->r = c.r;
		this->g = c.g;
		this->b = c.b;
	}

	static Color Lerp(Color a, Color b, double t)
	{
		return Color(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t);
	}
};

static Color NodeLowColor(255, 0, 0);
static Color NodeHighColor(0, 255, 0);
static Color WeightLowColor(0, 0, 255);
static Color WeightHighColor(255, 255, 0);
static Color BiasLowColor(0, 255, 255);
static Color BiasHighColor(255, 0, 255);
static Color RecurrentNodeLowColor(255, 255, 0);
static Color RecurrentNodeHighColor(255, 0, 255);

static Color NodeColor(double value)
{
	return Color::Lerp(NodeLowColor, NodeHighColor, Sigmoid().Activate(value));
}
static Color WeightColor(double value)
{
	return Color::Lerp(WeightLowColor, WeightHighColor, Sigmoid().Activate(value));
}
static Color BiasColor(double value)
{
	return Color::Lerp(BiasLowColor, BiasHighColor, Sigmoid().Activate(value));
}
static Color RecurrentNodeColor(double value)
{
	return Color::Lerp(RecurrentNodeLowColor, RecurrentNodeHighColor, Sigmoid().Activate(value));
}

static void SetColor(SDL_Renderer *renderer, Color c)
{
	SDL_SetRenderDrawColor(renderer, c.r, c.g, c.b, 255);
}

int main()
{
	// init sdl2
	SDL_Init(SDL_INIT_EVERYTHING);
	SDL_Window *window = SDL_CreateWindow("Recurrent Neural Network", 1920 / 3 * 2, SDL_WINDOWPOS_CENTERED, 1080, 920, SDL_WINDOW_SHOWN);
	SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	SDL_Event event;
	bool running = true;

	srand(time(NULL));
	RecurrentLayer p;
	p.Init(2, 2, new Sigmoid());
	MSE mse;

	// solve the AND problem
	std::vector<std::vector<double>> x = {{Zero, Zero}, {Zero, One}, {One, Zero}, {One, One}};
	std::vector<std::vector<double>> y = {{Zero, One}, {Zero, One}, {Zero, One}, {One, Zero}};

	// sdl loop
	int frame = 0;
	while (running)
	{
		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_QUIT)
			{
				running = false;
			}
		}
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		SDL_RenderClear(renderer);

		// train
		std::vector<double> o = p.Forward(x[frame % x.size()]);
		std::vector<double> dx = p.Backward(mse.Derivative(o, y[frame % y.size()]), .0001);

		// draw weights
		std::vector<std::vector<double>> w = p.GetWeights();
		for (int i = 0; i < w.size(); i++)
		{
			for (int j = 0; j < w[i].size(); j++)
			{
				SetColor(renderer, WeightColor((w[i][j] - .5) * 100));
				SDL_RenderDrawLine(renderer, 100 * i + 50, 100 * j + 50, 100 * i + 150, 100 * j + 150);
			}
		}

		// draw nodes
		std::vector<double> x = p.GetX();
		for (int i = 0; i < x.size(); i++)
		{
			SetColor(renderer, NodeColor((x[i] - .5) * 100));
			// circle
			SDL_Point center = {100 * i + 50, 50};
			SDL_RenderDrawPoint(renderer, center.x, center.y);
			for (int k = 0; k < 360; k++)
			{
				SDL_Point p = {center.x + 10 * cos(k * 3.14159265358979323846 / 180.0), center.y + 10 * sin(k * 3.14159265358979323846 / 180.0)};
				SDL_RenderDrawPoint(renderer, p.x, p.y);
			}
		}

		// draw recurrent nodes below the normal nodes in the same layer
		std::vector<double> pa = p.GetPreviousActivation();
		for (int i = 0; i < pa.size(); i++)
		{
			SetColor(renderer, RecurrentNodeColor((pa[i] - .5) * 100));
			// circle
			SDL_Point center = {100 * i + 50, 150};
			SDL_RenderDrawPoint(renderer, center.x, center.y);
			for (int k = 0; k < 360; k++)
			{
				SDL_Point p = {center.x + 10 * cos(k * 3.14159265358979323846 / 180.0), center.y + 10 * sin(k * 3.14159265358979323846 / 180.0)};
				SDL_RenderDrawPoint(renderer, p.x, p.y);
			}
		}

		// draw bias
		std::vector<double> b = p.GetBias();
		for (int i = 1; i < b.size() + 1; i++)
		{
			SetColor(renderer, BiasColor((b[i - 1] - .5) * 100));
			// small rect
			SDL_FRect rect = {100 * i + 50, 50, 50, 50};
			SDL_RenderFillRectF(renderer, &rect);
		}

		printf("\033[0;0H");

		printf("0 AND 0 = %f,%f\tShould Be: 0,1\n", p.Forward({Zero, Zero})[0], p.Forward({Zero, Zero})[1]);
		printf("0 AND 1 = %f,%f\tShould Be: 0,1\n", p.Forward({Zero, One})[0], p.Forward({Zero, One})[1]);
		printf("1 AND 0 = %f,%f\tShould Be: 0,1\n", p.Forward({One, Zero})[0], p.Forward({One, Zero})[1]);
		printf("1 AND 1 = %f,%f\tShould Be: 1,0\n", p.Forward({One, One})[0], p.Forward({One, One})[1]);

		SDL_RenderPresent(renderer);
		frame++;
	}

	return 0;
}
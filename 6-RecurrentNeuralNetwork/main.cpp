#include <iostream>
#include <thread>
#include <chrono>

#include "RecurrentNeuralNetwork.hpp"
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

double MeanSquaredError(std::vector<double> output, std::vector<double> target)
{
	double loss = 0;
	for (int i = 0; i < output.size(); i++)
	{
		loss += pow(output[i] - target[i], 2);
	}
	return loss;
}

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

// red
static Color NodeLowColor(255, 0, 0);
// green
static Color NodeHighColor(0, 255, 0);
// red
static Color WeightLowColor(255, 0, 0);
// green
static Color WeightHighColor(0, 255, 0);
// cyan
static Color BiasLowColor(0, 255, 255);
// magenta
static Color BiasHighColor(255, 0, 255);
// orange
static Color RecurrentNodeLowColor(255, 165, 0);
// magenta
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

static void DrawText(SDL_Renderer *renderer, TTF_Font *font, std::string text, int x, int y, SDL_Color color = {255, 255, 255}, double scale = 1.0)
{
	SDL_Surface *surfaceMessage = TTF_RenderText_Solid(font, text.c_str(), color);
	SDL_Texture *Message = SDL_CreateTextureFromSurface(renderer, surfaceMessage);
	SDL_Rect Message_rect;
	Message_rect.x = x;
	Message_rect.y = y;
	Message_rect.w = text.length() * 12 * scale;
	Message_rect.h = 24 * scale;
	SDL_RenderCopy(renderer, Message, NULL, &Message_rect);
	SDL_FreeSurface(surfaceMessage);
	SDL_DestroyTexture(Message);
}

int main()
{
	// init sdl2
	SDL_Init(SDL_INIT_EVERYTHING);
	TTF_Init();
	TTF_Font *font = TTF_OpenFont("Mono.ttf", 24);
	SDL_Window *window = SDL_CreateWindow("Recurrent Neural Network", 1920 / 3 * 2, SDL_WINDOWPOS_CENTERED, 1080, 920, SDL_WINDOW_SHOWN);
	SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	SDL_Event event;
	bool running = true;

	srand(time(NULL));
	RecurrentNeuralNetwork p;
	p.Init({2, 3, 2}, new Sigmoid(), -1, 1);
	MSE mse;

	// solve the AND problem
	std::vector<std::vector<double>> x = {{Zero, Zero}, {Zero, One}, {One, Zero}, {One, One}};
	std::vector<std::vector<double>> y = {{Zero, One}, {One, Zero}, {One, Zero}, {Zero, One}};

	double x_seperation = 200;
	double y_seperation = 50;

	// sdl loop
	int frame = 0;
	bool canTrain = true;
	while (running)
	{
		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_QUIT)
			{
				running = false;
			}

			if (event.type == SDL_KEYDOWN)
			{
				if (event.key.keysym.sym == SDLK_SPACE)
				{
					canTrain = false;
				}
			}
			if (event.type == SDL_KEYUP)
			{
				if (event.key.keysym.sym == SDLK_SPACE)
				{
					canTrain = true;
				}
			}
		}

		// weighted random
		// index 0,1,2 =.5; index 3=0.5

		if (canTrain)
		{
			std::vector<double> o = p.Forward(x[frame % 4]);
			double output = MeanSquaredError(o, y[frame % 4])*.001;
			std::vector<double> dx = p.Backward(mse.Derivative(o, y[frame % 4]), output);
		}
		else
		{
			for (int i = 0; i < 100; i++)
			{
				int index = rand() % 4;
				std::vector<double> o = p.Forward(x[index]);
				std::vector<double> dx = p.Backward(mse.Derivative(o, y[index]), .01);
			}
		}
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		SDL_RenderClear(renderer);

		// draw weights
		std::vector<std::vector<std::vector<double>>> w = p.GetWeights();
		for (int i = 0; i < w.size(); i++)
		{
			for (int j = 0; j < w[i].size(); j++)
			{
				for (int k = 0; k < w[i][j].size(); k++)
				{
					SetColor(renderer, WeightColor((w[i][j][k] - .5) * 100));
					SDL_RenderDrawLine(renderer, x_seperation * i + 50, y_seperation * j + 50, x_seperation * (i + 1) + 50, y_seperation * k + 50);
				}
			}
		}

		// draw nodes
		std::vector<std::vector<double>> x = p.GetX();
		for (int i = 0; i < x.size(); i++)
		{
			for (int j = 0; j < x[i].size(); j++)
			{
				SetColor(renderer, NodeColor((x[i][j] - .5) * 100));
				// circle
				SDL_Point center = {x_seperation * i + 50, y_seperation * j + 50};
				SDL_RenderDrawPoint(renderer, center.x, center.y);
				for (int k = 0; k < 360; k++)
				{
					SDL_Point p = {center.x + 10 * cos(k * 3.14159265358979323846 / 180.0), center.y + 10 * sin(k * 3.14159265358979323846 / 180.0)};
					SDL_RenderDrawPoint(renderer, p.x, p.y);
				}
			}
		}

		// draw recurrent nodes below the normal nodes in the same layer
		std::vector<std::vector<double>> pa = p.GetPreviousActivation();
		for (int i = 0; i < pa.size(); i++)
		{
			for (int j = x[i].size(); j < pa[i].size() + x[i].size(); j++)
			{
				SetColor(renderer, RecurrentNodeColor((pa[i][j - (x[i].size() - 1)] - .5) * 100));
				// circle
				SDL_Point center = {x_seperation * i + 50, y_seperation * j + 50};
				SDL_RenderDrawPoint(renderer, center.x, center.y);
				for (int k = 0; k < 360; k++)
				{
					SDL_Point p = {center.x + 10 * cos(k * 3.14159265358979323846 / 180.0), center.y + 10 * sin(k * 3.14159265358979323846 / 180.0)};
					SDL_RenderDrawPoint(renderer, p.x, p.y);
				}
			}
		}

		// draw bias
		std::vector<std::vector<double>> b = p.GetBias();
		for (int i = 1; i < b.size() + 1; i++)
		{
			for (int j = 0; j < b[i - 1].size(); j++)
			{
				SetColor(renderer, BiasColor((b[i - 1][j] - .5) * 100));
				// small rect
				SDL_FRect rect = {x_seperation * i + 50 - 5, y_seperation * j + 50 - 5, 10, 10};
				SDL_RenderFillRectF(renderer, &rect);
			}
		}

		// render node values
		for (int i = 0; i < x.size(); i++)
		{
			for (int j = 0; j < x[i].size(); j++)
			{
				DrawText(renderer, font, std::to_string(x[i][j]), x_seperation * i + 50, y_seperation * j + 50);
			}
		}

		// render recurrent node values
		for (int i = 0; i < pa.size(); i++)
		{
			for (int j = x[i].size(); j < pa[i].size() + x[i].size(); j++)
			{
				DrawText(renderer, font, std::to_string(pa[i][j - (x[i].size())]), x_seperation * i + 50, y_seperation * j + 50);
			}
		}

		// render tests
		int pos_x = 0;
		int pos_y = 1080 - 260;
		DrawText(renderer, font, "0 AND 0 = " + std::to_string(p.Forward({Zero, Zero})[0]) + "," + std::to_string(p.Forward({Zero, Zero})[1]) + "\tShould Be: " + std::to_string(y[0][0]) + "," + std::to_string(y[0][1]), pos_x, pos_y);
		DrawText(renderer, font, "0 AND 1 = " + std::to_string(p.Forward({Zero, One})[0]) + "," + std::to_string(p.Forward({Zero, One})[1]) + "\tShould Be: " + std::to_string(y[1][0]) + "," + std::to_string(y[1][1]), pos_x, pos_y + 24);
		DrawText(renderer, font, "1 AND 0 = " + std::to_string(p.Forward({One, Zero})[0]) + "," + std::to_string(p.Forward({One, Zero})[1]) + "\tShould Be: " + std::to_string(y[2][0]) + "," + std::to_string(y[2][1]), pos_x, pos_y + 24 * 2);
		DrawText(renderer, font, "1 AND 1 = " + std::to_string(p.Forward({One, One})[0]) + "," + std::to_string(p.Forward({One, One})[1]) + "\tShould Be: " + std::to_string(y[3][0]) + "," + std::to_string(y[3][1]), pos_x, pos_y + 24 * 3);

		SDL_RenderPresent(renderer);
		frame++;
	}

	return 0;
}
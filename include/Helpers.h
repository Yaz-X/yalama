#pragma once

#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <string>
#include <cctype>

static std::string ResolvePath(std::string path)
{
    if (!path.empty() && path[0] == '~')
    {
        const char *home = std::getenv("HOME");

        if (!home)
            home = std::getenv("USERPROFILE"); // Windows fallback

        if (home)
            path = std::string(home) + path.substr(1);
        else
            throw std::runtime_error("Cannot resolve '~' because HOME is not set");
    }

    return path;
}

static std::string ToLower(const std::string &input)
{
    std::string result = input;

    std::transform(
        result.begin(),
        result.end(),
        result.begin(),
        [](unsigned char c)
        {
            return static_cast<char>(std::tolower(c));
        });

    return result;
}

static std::string Trim(const std::string &input)
{
    const auto start = input.find_first_not_of(" \t\n\r");
    const auto end = input.find_last_not_of(" \t\n\r");
    std::string output = "";

    if (start != std::string::npos)
        output = input.substr(start, end - start + 1);

    return output;
}

static std::string TrimToLower(const std::string &input)
{
    std::string trimmed = Trim(input);
    trimmed = ToLower(trimmed);

    return trimmed;
}

static int GetConsoleWidth()
{
    int width = 120;

#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
    {
        width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    }
#else
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0)
    {
        width = w.ws_col;
    }
#endif

    return width;
}

static void ShowProgressBar(float progress)
{
    int consoleWidth = GetConsoleWidth();

    int barWidth = consoleWidth - 20;

    if (barWidth < 10)
    {
        barWidth = 10;
    }

    int pos = static_cast<int>(barWidth * progress);

    std::cout << "\r";

    for (int i = 0; i < barWidth; ++i)
    {
        if (i <= pos)
        {
            std::cout << "█";
        }
        else
        {
            std::cout << "░";
        }
    }

    std::cout << " " << static_cast<int>(progress * 100.0f) << "%";

    std::cout.flush();
}

static std::string Replace(const std::string &input, const std::string &oldValue, const std::string &newValue)
{
    std::string result = input;

    size_t pos = result.find(oldValue);

    if (pos != std::string::npos)
        result.replace(pos, oldValue.length(), newValue);

    return result;
}
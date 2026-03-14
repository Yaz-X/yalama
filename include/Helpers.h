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

static void ReplaceAll(std::string &str, const std::string &from, const std::string &to)
{
    size_t pos = 0;

    while ((pos = str.find(from, pos)) != std::string::npos)
    {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
}

static void CleanIncompleteUTF8(std::string &s)
{
    if (s.empty())
        return;

    int len = s.size();
    int check = std::min(4, len);

    for (int i = 1; i <= check; i++)
    {
        unsigned char c = s[len - i];

        // ASCII
        if ((c & 0x80) == 0)
            return;

        // start of UTF8 sequence
        if ((c & 0xC0) == 0xC0)
        {
            int expected = 0;

            if ((c & 0xE0) == 0xC0)
                expected = 2;
            else if ((c & 0xF0) == 0xE0)
                expected = 3;
            else if ((c & 0xF8) == 0xF0)
                expected = 4;

            int actual = i;

            if (actual < expected)
                s.resize(len - actual);

            return;
        }
    }
}

static bool IsValidUTF8(const std::string& s)
{
    int remaining = 0;

    for (unsigned char c : s)
    {
        if (remaining == 0)
        {
            if ((c >> 7) == 0b0)          // 1 byte (ASCII)
                continue;
            else if ((c >> 5) == 0b110)   // 2 bytes
                remaining = 1;
            else if ((c >> 4) == 0b1110)  // 3 bytes
                remaining = 2;
            else if ((c >> 3) == 0b11110) // 4 bytes
                remaining = 3;
            else
                return false;
        }
        else
        {
            if ((c >> 6) != 0b10)
                return false;

            remaining--;
        }
    }

    return remaining == 0;
}
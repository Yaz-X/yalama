#include <filesystem>
#include <fstream>
#include <string>

class RuntimeEnvironment
{
public:
    static bool IsDocker()
    {
        bool isDocker = false;

        if (std::filesystem::exists("/.dockerenv"))
        {
            isDocker = true;
        }
        else if (std::filesystem::exists("/run/.containerenv"))
        {
            isDocker = true;
        }

        return isDocker;
    }
};
#include <iostream>
// #include <memory>
// #include <cassert>
using namespace std;

template<typename F>
constexpr auto stars(const F& func) {
    return [func](auto&&... args) {
        cout << "*******" << endl;
        func(forward<decltype(args)>(args)...);
        cout << "\n*******" << endl;
    };
}

void hello_impl(string xx, string yy) {
   cout << "hello, world! " << xx << yy << endl;
}

constexpr auto hello = stars(hello_impl);

int main() {
    // hello("aa", "bb");

    return 0;
}
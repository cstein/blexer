main() {
    extrn putnum;
    extrn putchar;
    auto a, b, c;
    a = 0;
    b = 1;
    while (a < 10000) {
        putnum(a);
        putchar(10);
        c = a + b;
        a = b;
        b = c;
    }
}

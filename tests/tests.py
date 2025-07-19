from main import BinaryOperator, UnaryOperator, Value, Variable


def test_binary_operator():
    bin_op = BinaryOperator(op="+")
    x, y = 1, 1
    res = bin_op.forward(x, y)
    assert res == 2

    bin_op = BinaryOperator(op="-")
    res = bin_op.forward(x, y)
    assert res == 0


def test_unary_operator():
    un_op = UnaryOperator("sin")
    x = 0
    res = un_op.forward(x)
    assert res == 0.0

    un_op = UnaryOperator("cos")
    res = un_op.forward(x)
    assert res == 1.0

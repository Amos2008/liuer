
class B:
  i = 10
  def test(self):
    print(self.i)

b = B()

b.test()


# c++ void test(this) {
#         std::cout << this->i << endl;
#     }  默认传入一个非成员变量，在使用对象调用的时候将对象的地址传递进去，在成员函数内部对类内部的成员都会添加上this->调用

class A:
    pass


a = A()
print(type(A))
print(type(a))
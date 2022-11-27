#include <cstddef>
#include <string>
#include <vector>

using namespace ROOT;
using namespace TMVA::Experimental;

template <size_t NCols, typename First, typename... Rest>
class HelperFunc
{
  static_assert(1 + sizeof...(Rest) == NCols, "");

private:
  // For N = 0, 1, ..., NCols - 1
  template <size_t N>
  struct AssignToTensor
  {
    template <typename FirstArgs, typename... RestArgs>
    static void Call(TMVA::Experimental::RTensor<float> &x,
                     const size_t offset,
                     FirstArgs &&first, RestArgs &&...rest)
    {
      // Assign x[offset + N] = first
      x.GetData()[offset + N] = first;
      // Assign x[offset + N - 1] = first element of rest...
      AssignToTensor<N + 1>::Call(x, offset, std::forward<RestArgs>(rest)...);
    }
  };
  // Stop at N = NCols, do nothing
  template <>
  struct AssignToTensor<NCols>
  {
    template <typename... Args>
    static void Call(TMVA::Experimental::RTensor<float> &, const size_t offset,
                     Args...) {}
  };

private:
  size_t offset = 0;
  TMVA::Experimental::RTensor<float> &fTensor;

public:
  HelperFunc(TMVA::Experimental::RTensor<float> &x_tensor)
      : fTensor(x_tensor) {}

  void operator()(First first, Rest... rest)
  {
    AssignToTensor<0>::Call(fTensor, offset, std::forward<First>(first), std::forward<Rest>(rest)...);
    // offset += NCols;
  };
};

class Generator_t
{
private:
  size_t i = 0;

public:
  RTensor<float> operator()(const size_t batch_size, RDataFrame &x_rdf,
                            const size_t nevt)
  {

    // TO DO: make column input dynamic with std::make_index_sequence
    std::vector<std::string> cols = {"jet1_phi", "jet1_eta", "jet1_pt", "jet2_phi", "jet2_pt"};
    TMVA::Experimental::RTensor<float> x_tensor({nevt, 5});
    HelperFunc<5, float &, float &, float &, float &, float &> func(x_tensor);
    size_t offset = 0;

    x_rdf.Foreach(func, cols);

    auto data_len = x_tensor.GetShape()[0];
    auto num_column = x_tensor.GetShape()[1];
    // std::cout << "data len = " << data_len << " and num_column = " << num_column
    //           << std::endl;
    // std::cout << "Rtensor = \n";
    // std::cout << x_tensor << std::endl;

    if (i + batch_size < data_len)
    {
      unsigned long offset = i * num_column;
      RTensor<float> x_batch(x_tensor.GetData() + offset,
                             {batch_size, num_column});

      i += batch_size;
      // cout << "x_batch is: " << x_batch << "\n"<< "\n";

      return x_batch;
    }
    else
    {
      return x_tensor.Slice({{0, 0}, {0, 0}});
    }
  }
};

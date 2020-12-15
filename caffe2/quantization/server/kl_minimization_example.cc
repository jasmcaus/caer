#include "caffe2/core/logging.h"
#include "kl_minimization.h"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
using namespace dnnlowp;

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    cerr << "Usage: " << argv[0] << " in_file out_file [preserve_sparsity]"
         << endl;
    return -1;
  }

  ifstream in(argv[1]);
  ofstream out(argv[2]);
  bool preserve_sparsity = argc >= 4 ? atoi(argv[3]) : false;

  string line;
  while (getline(in, line)) {
    istringstream ist(line);

    int op_index, output_index;
    string op_type, tensor_name;
    float min, max;
    int nbins;

    ist >> op_index >> op_type >> output_index >> tensor_name >> min >> max >>
        nbins;

    vector<uint64_t> bins;
    for (int i = 0; i < nbins; ++i) {
      uint64_t cnt;
      ist >> cnt;
      bins.push_back(cnt);
    }
    assert(bins.size() == nbins);

    Histogram hist = Histogram(min, max, bins);
    TensorQuantizationParams qparams =
        KLDivergenceMinimization().ChooseQuantizationParams(
            hist, preserve_sparsity);

    out << op_index << " " << op_type << " " << output_index << " "
        << tensor_name << " " << qparams.Min() << " " << qparams.Max() << endl;
  }

  return 0;
}

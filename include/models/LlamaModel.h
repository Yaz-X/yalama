#pragma once

#include "ModelBase.h"

class LlamaModel : public ModelBase
{

protected:
  
  void PopulateWeightNames() override;

public:
    LlamaModel();
  };

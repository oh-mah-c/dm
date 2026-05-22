#include <stdio.h>
#include <stdlib.h>
#include "tensorflow/c/c_api.h"

int main() {
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* opts = TF_NewSessionOptions();
    const char* tags[] = {"serve"};
    TF_Session* session = TF_LoadSessionFromSavedModel(opts, NULL, "test_trainable_model", tags, 1, graph, NULL, status);

    if (TF_GetCode(status) != TF_OK) {
        printf("Error loading SavedModel: %s\n", TF_Message(status));
        return 1;
    }
    printf("Successfully loaded SavedModel!\n");

    size_t pos = 0;
    TF_Operation* op;
    int count = 0;
    while ((op = TF_GraphNextOperation(graph, &pos)) != NULL) {
        printf("Op %d: name='%s' type='%s'\n", ++count, TF_OperationName(op), TF_OperationOpType(op));
    }

    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(opts);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    return 0;
}

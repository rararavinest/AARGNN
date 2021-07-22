import torch
import time
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter as Param
from torch.nn import Parameter
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul, masked_select_nnz
from torch_geometric.nn.conv import MessagePassing

import numpy as np
from torch.autograd import Variable



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def getrel(num_nodes):
    num_atoms=num_nodes
    off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)
    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)
    return  rel_rec,rel_send

def repeat(gloabl_attr, target_attr, axis=0):
    """Repeats a `Tensor`'s elements along an axis by custom amounts.
    Args:
        gloabl_attr: a 1D sequence of numer of repeats
        target_attr: tensor to repeat
        axi s: to repeat along
    :return: repeated tensor
    """
    #get the repeat para
    node_num=target_attr.shape[0]
    broadcast_global=gloabl_attr.repeat(node_num,1)
    return broadcast_global

def aggregate(input,global_attr):
    #aggregate to global
    #mean
    #input shape node:[node_num,in_dims] edges:[num_node,]
    g_attr_m=torch.mean(input,1,True)
    return g_attr_m

def edgeindex2attr(edgeindex):
    #[(n-1)^2,n_hid]=>[n,n,n_hid]
    return attr

def edgeattr2index(adj):
    return index


class RGCNConv3(torch.nn.Module):
    #rewrite RGCNConv as 3 attrs
    """
    orginal: 2 attrs
    rewrite 3 attrs
    process x=>h
    input x[in_channels],edge[number_relations], global[number_lobals]
    output target[out_channels]
    parameter: in_channels, out_channels, num_relations, num_globals
    """
    def __init__(self, in_channels, hidden_unit, out_channels, num_relations, num_globals, **kwargs):
        super(RGCNConv3, self).__init__()
        self.in_channels = in_channels
        self.hidden_unit = hidden_unit
        self.out_channels = out_channels

    def all2edges(self,node_attr,edge_attr,global_attr):
        #update edges
        # node:[num_times,num_node,num_feature]
        # edge:[num_]
        # node2edge
        #sender2edges
        #reciver2edges
        num_nodes=node_attr.shape[0]
        rel_rec,rel_send=getrel(num_nodes)
        x1=node_attr
        receivers = torch.matmul(rel_rec, x1)
        senders = torch.matmul(rel_send, x1)
        edges_n = torch.cat([senders, receivers], dim=-1)
        #global2edges
        x2=edge_attr
        edges_g =torch.matmul(global_attr,x2)
        #aggregate all
        edges=torch.cat([edges_n,edges_g],dim=-1)
        return edges

    def all2globals(self,node_attr,edge_attr,global_attr):
        #update globals
        #node2g
        g_attr_n = torch.mean(node_attr, 1, True)
        #global_n=aggregate(node_attr,global_attr)
        #edge2g
        g_attr_e = torch.mean(edge_attr, 1, True)
        #global_e=aggregate(edge_attr,global_attr)
        globals=torch.cat([g_attr_e,g_attr_n],dim=0)
        return globals

    def all2nodes(self,node_attr):
        #edge2node

        #global2node


        return node_attr


    def forward(self, inputs, edge_attr, global_attr):
        #the input contains: x, e, g
        #output tensor:size of out_channels
        x_n = inputs
        # update edges
        edges=self.all2edges(x_n,edge_attr,global_attr)
        # update global
        globals=self.all2globals(x_n,edges,global_attr)
        #update nodes
        nodes=self.all2nodes(x_n,edges,globals)
        #a mlp
        output=nodes
        # output x'
        return output


class RGCNConv3_xi(torch.nn.Module):
    #rewrite RGCNConv as 3 attrs
    """
    orginal: 2 attrs
    rewrite 3 attrs
    process x=>h
    input x[num_node * in_channels],edge[number_relations * features], global[number_globals]
    output target[out_channels]
    parameter: in_channels, out_channels
    """
    def __init__(self, in_channels, hidden_unit, out_channels, **kwargs):
        super(RGCNConv3_xi, self).__init__()
        self.in_channels = in_channels
        self.hidden_unit = hidden_unit
        self.out_channels = out_channels

    def all2edges(self,node_attr,edge_attr,global_attr):
        #update edges
        # node:[num_node,num_feature]
        # edge:[num_node*num_node, feature]
        # node2edge
        #sender2edges
        #reciver2edges
        num_nodes=node_attr.shape[0]
        rel_rec,rel_send=getrel(num_nodes)
        x1=node_attr
        receivers = torch.matmul(rel_rec, x1)
        senders = torch.matmul(rel_send, x1)
        edges_n = torch.cat([senders, receivers], dim=-1)
        #global2edges
        x2=edge_attr
        edges_g =torch.matmul(global_attr,x2)
        #aggregate all
        edges=torch.cat([edges_n,edges_g],dim=-1)
        return edges

    def all2globals(self,node_attr,edge_attr,global_attr):
        #update globals
        #node2g
        g_attr_n = torch.mean(node_attr, 1, True)
        #global_n=aggregate(node_attr,global_attr)
        #edge2g
        g_attr_e = torch.mean(edge_attr, 1, True)
        #global_e=aggregate(edge_attr,global_attr)
        globals=torch.cat([g_attr_e,g_attr_n],dim=0)
        return globals

    def all2nodes(self,node_attr):
        #edge2node
        #global2node


        return node_attr


    def forward(self, inputs, edge_attr, global_attr):
        #the input contains: x, e, g
        #output tensor:size of out_channels
        x_n = inputs
        # update edges
        edges=self.all2edges(x_n,edge_attr,global_attr)
        # update global
        globals=self.all2globals(x_n,edges,global_attr)
        #update nodes
        nodes=self.all2nodes(x_n,edges,globals)
        #a mlp
        output=nodes
        # output x'
        return output






class RRGCN(torch.nn.Module):
    """
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases.

        nodes:[num_node, in_dims]
        edges:[num_node,num_node,num_edges]
        global:[num_globals]
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_unit,
                 num_relations: int, num_globals, num_bases: int):
        super(RRGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_unit = hidden_unit
        self.num_relations = num_relations
        self.num_globals = num_globals
        self._create_layers()


    def _create_input_gate_layers(self):

        self.conv_x_i = RGCNConv3_xi(in_channels=self.in_channels,
                                 out_channels=self.hidden_unit)

        self.conv_h_i = RGCNConv3(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 num_relations=self.num_relations,
                                 num_globals=self.num_globals)


    def _create_forget_gate_layers(self):

        self.conv_x_f = RGCNConv3(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 num_relations=self.num_relations,
                                 num_globals=self.num_globals,
                                 num_bases=self.num_bases)

        self.conv_h_f = RGCNConv3(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 num_relations=self.num_relations,
                                 num_globals=self.num_globals,
                                 num_bases=self.num_bases)


    def _create_cell_state_layers(self):

        self.conv_x_c = RGCNConv3(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 num_relations=self.num_relations,
                                 num_globals=self.num_globals,
                                 num_bases=self.num_bases)

        self.conv_h_c = RGCNConv3(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 num_relations=self.num_relations,
                                 num_globals=self.num_globals,
                                 num_bases=self.num_bases)


    def _create_output_gate_layers(self):

        self.conv_x_o = RGCNConv3(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 num_relations=self.num_relations,
                                 num_globals=self.num_globals,
                                 num_bases=self.num_bases)

        self.conv_h_o = RGCNConv3(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 num_relations=self.num_relations,
                                 num_globals=self.num_globals,
                                 num_bases=self.num_bases)


    def _create_layers(self):
        self._create_input_gate_layers()
        self._create_forget_gate_layers()
        self._create_cell_state_layers()
        self._create_output_gate_layers()


    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H


    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels)
        return C


    def _calculate_input_gate(self, X, edge_attr, global_attr, H, C):
        I = self.conv_x_i(X, edge_attr, global_attr)
        I = I + self.conv_h_i(H, edge_attr, global_attr)
        I = torch.sigmoid(I)
        return I


    def _calculate_forget_gate(self, X, edge_attr, global_attr, H, C):
        F = self.conv_x_f(X, edge_attr, global_attr)
        F = F + self.conv_h_f(H, edge_attr, global_attr)
        F = torch.sigmoid(F)
        return F


    def _calculate_cell_state(self, X, edge_attr, global_attr, H, C, I, F):
        T = self.conv_x_c(X, edge_attr, global_attr)
        T = T + self.conv_h_c(H, edge_attr, global_attr)
        T = torch.tanh(T)
        C = F*C + I*T
        return C


    def _calculate_output_gate(self, X, edge_attr, global_attr, H, C):
        O = self.conv_x_o(X, edge_attr, global_attr)
        O = O + self.conv_h_o(H, edge_attr, global_attr)
        O = torch.sigmoid(O)
        return O


    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H


    def forward(self, X, edge_attr, global_attr, H, C):
        """
        Making a forward pass. If the hidden state and cell state matrices are
        not present when the forward pass is called these are initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node attributes.
            * **edge_attr** *(PyTorch Long Tensor)* - Graph edge attributes.
            * **global_attr** *(PyTorch Long Tensor)* - global attr.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_attr, global_attr,  H, C)
        F = self._calculate_forget_gate(X, edge_attr, global_attr, H, C)
        C = self._calculate_cell_state(X, edge_attr, global_attr, H, C, I, F)
        O = self._calculate_output_gate(X, edge_attr, global_attr, H, C)
        H = self._calculate_hidden_state(O, C)
        return H, C

    def loop(self, input_x, input_e, input_g):
        batch_size = input_x.size(0)
        time_step = input_x.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        for i in range(time_step):
            Hidden_State, Cell_State = self.forward(torch.squeeze(input_x[:, i:i + 1, :]), Hidden_State, Cell_State)
        return Hidden_State, Cell_State

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_unit).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_unit).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_unit))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_unit))
            return Hidden_State, Cell_State




















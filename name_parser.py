# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import autodiff as ad
import ply.yacc as yacc
import ply.lex as lex


class AutodiffParser:
    @staticmethod
    def parse(name, inputs):
        '''
        Parameters
        ----------
        name: str of the graph.
        inputs: the terminal nodes.
        ----------
        
        Returns:
        ----------
        The constructed graph.
        ----------
        '''
        inputs_map = dict(zip([i.name for i in inputs], inputs))

        reserved = {
            'T.einsum': 'EINSUM_PREFIX',
            'T.tensorinv': 'EINSUM_INVERSE',
            'T.identity': 'IDENTITY',
            'ind=': 'INV_INDEX',
        }
        # Parsing rules
        tokens = [
            'NAME',
            'NUMBER',
            'PLUS',
            'MINUS',
            'TIMES',
            'LPAREN',
            'RPAREN',
            'EINSUM_SUBSCRIPT',
            'COMMA',
            'ID',
        ] + list(reserved.values())

        def t_ID(t):
            r'[a-zA-Z_]*[\.\=][a-zA-Z]*'
            t.type = reserved.get(t.value, 'ID')  # Check for reserved words
            return t

        # Tokens

        t_PLUS = r'\+'
        t_MINUS = r'-'
        t_TIMES = r'\*'
        t_LPAREN = r'\('
        t_RPAREN = r'\)'
        t_NUMBER = r'[0-9]+\.?[0-9]*'
        t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'
        t_EINSUM_SUBSCRIPT = r'\'[a-zA-Z,]*->[a-zA-Z]*\''
        t_COMMA = r'\,'

        # Ignored characters
        t_ignore = " \t"

        def t_newline(t):
            r'\n+'
            t.lexer.lineno += t.value.count("\n")

        def t_error(t):
            print("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)

        # Build the lexer
        lexer = lex.lex()

        precedence = (
            ('left', 'PLUS', 'MINUS'),
            ('left', 'TIMES'),
        )

        def p_statement_expr(t):
            'statement : expression'
            t[0] = t[1]

        def p_expression_binop(t):
            '''expression : expression PLUS expression
                          | expression MINUS expression
                          | expression TIMES expression
                          '''
            if t[2] == '+':
                t[0] = t[1] + t[3]
            elif t[2] == '-':
                t[0] = t[1] - t[3]
            elif t[2] == '*':
                t[0] = t[1] * t[3]

        def p_expression_group(t):
            'expression : LPAREN expression RPAREN'
            t[0] = t[2]

        def p_expression_number(t):
            'expression : NUMBER'
            t[0] = float(t[1])

        def p_expression_name(t):
            'expression : NAME'
            try:
                t[0] = inputs_map[t[1]]
            except LookupError:
                print("Undefined name '%s'" % t[1])
                t[0] = 0

        def p_expression_tensorinv(t):
            'expression : EINSUM_INVERSE LPAREN expression COMMA INV_INDEX NUMBER RPAREN'
            t[0] = ad.tensorinv(t[3], ind=int(t[6]))

        def p_expression_identity(t):
            'expression : IDENTITY LPAREN NUMBER RPAREN'
            t[0] = ad.identity(t[3])

        def p_expression_einsum(t):
            'expression : EINSUM_PREFIX LPAREN EINSUM_SUBSCRIPT INPUTS RPAREN'
            # Below we get rid of the quotation marks.
            t[0] = ad.einsum(t[3][1:-1], *t[4])

        def p_expression_einsum_inputs(t):
            '''
            INPUTS : COMMA expression INPUTS
                   | COMMA expression 
            '''
            t[0] = [t[2]]
            if len(t) == 4:
                t[0] += t[3]

        def p_error(t):
            print("Syntax error at '%s'" % t.value)

        parser = yacc.yacc()

        return parser.parse(name)

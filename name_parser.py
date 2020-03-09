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

        # Parsing rules
        tokens = (
            'NAME',
            'PLUS',
            'MINUS',
            'TIMES',
            'LPAREN',
            'RPAREN',
        )

        # Tokens

        t_PLUS = r'\+'
        t_MINUS = r'-'
        t_TIMES = r'\*'
        t_LPAREN = r'\('
        t_RPAREN = r'\)'
        t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'

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

        def p_expression_name(t):
            'expression : NAME'
            try:
                t[0] = inputs_map[t[1]]
            except LookupError:
                print("Undefined name '%s'" % t[1])
                t[0] = 0

        def p_error(t):
            print("Syntax error at '%s'" % t.value)

        parser = yacc.yacc()

        return parser.parse(name)

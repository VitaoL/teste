-- Configuração de codificação para evitar erros de acentuação
SET CLIENT_ENCODING TO 'UTF8';

-- 1. LIMPEZA TOTAL (Apaga tabelas antigas para recriar do zero)
DROP TABLE IF EXISTS mensagem CASCADE;
DROP TABLE IF EXISTS avaliacao CASCADE;
DROP TABLE IF EXISTS pagamento CASCADE;
DROP TABLE IF EXISTS aula CASCADE;
DROP TABLE IF EXISTS disponibilidade CASCADE;
DROP TABLE IF EXISTS curso CASCADE;
DROP TABLE IF EXISTS professor_estilo CASCADE;
DROP TABLE IF EXISTS professor_instrumento CASCADE;
DROP TABLE IF EXISTS status_aula CASCADE;
DROP TABLE IF EXISTS estilo_musical CASCADE;
DROP TABLE IF EXISTS instrumento CASCADE;
DROP TABLE IF EXISTS professor CASCADE;
DROP TABLE IF EXISTS aluno CASCADE;

-- 2. CRIAÇÃO DAS TABELAS

-- Tabelas Auxiliares (Catálogos)
CREATE TABLE instrumento (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE estilo_musical (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE status_aula (
    id INTEGER PRIMARY KEY,
    nome VARCHAR(50) NOT NULL UNIQUE
);

-- Usuários
CREATE TABLE aluno (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    senha_hash VARCHAR(255) NOT NULL,
    telefone VARCHAR(20),
    foto_url TEXT,
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE professor (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    senha_hash VARCHAR(255) NOT NULL,
    biografia TEXT,
    valor_base_aula DECIMAL(10, 2) NOT NULL,
    foto_url TEXT,
    link_video_demo TEXT,
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Relacionamentos N:N
CREATE TABLE professor_instrumento (
    professor_id INTEGER REFERENCES professor(id) ON DELETE CASCADE,
    instrumento_id INTEGER REFERENCES instrumento(id) ON DELETE CASCADE,
    PRIMARY KEY (professor_id, instrumento_id)
);

CREATE TABLE professor_estilo (
    professor_id INTEGER REFERENCES professor(id) ON DELETE CASCADE,
    estilo_id INTEGER REFERENCES estilo_musical(id) ON DELETE CASCADE,
    PRIMARY KEY (professor_id, estilo_id)
);

-- Produtos (Cursos/Gigs)
CREATE TABLE curso (
    id SERIAL PRIMARY KEY,
    professor_id INTEGER REFERENCES professor(id) ON DELETE CASCADE,
    titulo VARCHAR(100) NOT NULL,
    descricao TEXT NOT NULL,
    preco DECIMAL(10, 2) NOT NULL,
    video_url TEXT,
    thumbnail_url TEXT,
    modalidade VARCHAR(20) DEFAULT 'online',
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agenda
CREATE TABLE disponibilidade (
    id SERIAL PRIMARY KEY,
    professor_id INTEGER REFERENCES professor(id) ON DELETE CASCADE,
    dia_semana INTEGER NOT NULL, -- 0=Dom, 1=Seg, etc.
    hora_inicio TIME NOT NULL,
    hora_fim TIME NOT NULL
);

-- Transações (Aulas e Pagamentos)
CREATE TABLE aula (
    id SERIAL PRIMARY KEY,
    aluno_id INTEGER REFERENCES aluno(id) ON DELETE SET NULL,
    professor_id INTEGER REFERENCES professor(id) ON DELETE SET NULL,
    curso_id INTEGER REFERENCES curso(id) ON DELETE SET NULL,
    data_hora TIMESTAMP NOT NULL,
    duracao_minutos INTEGER DEFAULT 60,
    status_id INTEGER REFERENCES status_aula(id),
    modalidade VARCHAR(20),
    valor_acordado DECIMAL(10, 2) NOT NULL,
    link_reuniao TEXT,
    endereco_aula TEXT,
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE pagamento (
    id SERIAL PRIMARY KEY,
    aula_id INTEGER UNIQUE REFERENCES aula(id) ON DELETE CASCADE,
    valor DECIMAL(10, 2) NOT NULL,
    status VARCHAR(20),
    metodo VARCHAR(50),
    data_pagamento TIMESTAMP,
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Social
CREATE TABLE avaliacao (
    id SERIAL PRIMARY KEY,
    aula_id INTEGER UNIQUE REFERENCES aula(id) ON DELETE CASCADE,
    aluno_id INTEGER REFERENCES aluno(id),
    professor_id INTEGER REFERENCES professor(id),
    nota INTEGER CHECK (nota >= 1 AND nota <= 5),
    comentario TEXT,
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE mensagem (
    id SERIAL PRIMARY KEY,
    aula_id INTEGER REFERENCES aula(id) ON DELETE CASCADE,
    remetente_tipo VARCHAR(10),
    texto TEXT NOT NULL,
    lida BOOLEAN DEFAULT FALSE,
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
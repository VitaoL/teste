-- Configuração de codificação (Garante que acentos funcionem)
SET CLIENT_ENCODING TO 'UTF8';

-- 1. STATUS E CATEGORIAS
INSERT INTO status_aula (id, nome) VALUES (1, 'SOLICITADA'), (2, 'CONFIRMADA'), (3, 'CANCELADA_ALUNO'), (4, 'CANCELADA_PROFESSOR'), (5, 'CONCLUIDA') ON CONFLICT (id) DO NOTHING;

INSERT INTO instrumento (nome) VALUES 
('Violão'), ('Guitarra'), ('Piano'), ('Teclado'), ('Bateria'), ('Violino'), ('Canto'), ('Baixo'), ('Voz'), 
('Produção Musical'), ('Loop Pedal'), ('Violoncelo'), ('Saxofone'), ('Ukulele'), ('Teoria Musical') 
ON CONFLICT (nome) DO NOTHING;

INSERT INTO estilo_musical (nome) VALUES 
('Rock'), ('Pop'), ('Clássico'), ('Jazz'), ('MPB'), ('Sertanejo'), ('Gospel'), ('Blues'), ('Metal'), 
('R&B'), ('Axé'), ('Folk'), ('Soul'), ('Funk'), ('Bossa Nova'), ('Ópera'), ('Country') 
ON CONFLICT (nome) DO NOTHING;

-- 2. ALUNOS
-- Senha para todos: 123
INSERT INTO aluno (nome, email, senha_hash, telefone) VALUES 
('Bruna Kutova', 'bruna@gmail.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', '31999998888'),
('Rafael Souza', 'rafael@gmail.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', '31988887777'),
('Lucas Silva', 'lucas@gmail.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', '31977776666'),
('Mariana Costa', 'mariana@gmail.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', '31966665555'),
('Pedro Santos', 'pedro@gmail.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', '31955554444')
ON CONFLICT (email) DO NOTHING;

-- 3. PROFESSORES (DREAM TEAM)
INSERT INTO professor (nome, email, senha_hash, valor_base_aula, biografia, link_video_demo, foto_url) VALUES 
-- ORIGINAIS
('Daniel Augusto', 'daniel.augusto@gmail.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 120.00, 'Doutor em Música. Ensino desde a teoria básica até a prática avançada.', 'https://www.youtube.com/embed/2t7s5FZZwXo', 'https://verboaria.com.br/wp-content/uploads/2024/02/Maestro-Daniel-Augusto-puc-minas-1-780x470.jpeg'),
('Melina Peixoto', 'mel.peixoto@gmail.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 150.00, 'Cantora Lírica e Fonoaudióloga. Especialista em saúde vocal e técnica Bel Canto.', 'https://www.youtube.com/embed/a7zhAlhABrE', 'https://yt3.googleusercontent.com/ytc/AIdro_nlJn_QGZKL15cOWYFWUO0vtR53gDnCEsRHdXxhYrOim38=s900-c-k-c0x00ffffff-no-rj'),
('João Barone', 'joao.rock@gmail.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 90.00, 'Baterista dos Paralamas do Sucesso. Foco em pegada e groove brasileiro.', 'https://www.youtube.com/embed/sHI8DLpO2ag', 'https://boomerangmusic.com.br/wp-content/uploads/2024/08/mini-Joao-Barone-Foto-2bx-credito-Marcio-Farias.jpg'),

-- CLÁSSICOS & ÓPERA
('Andrea Bocelli', 'andrea@bocelli.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 3000.00, 'Tenor italiano mundialmente famoso. Ensino a arte de emocionar através da voz.', 'https://www.youtube.com/embed/4RJqcTsXX-8', 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Andrea_Bocelli_2019.jpg/640px-Andrea_Bocelli_2019.jpg'),
('Renée Fleming', 'renee@soprano.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 2800.00, 'Soprano aclamada. Masterclass em interpretação operística e controle de respiração.', 'https://www.youtube.com/embed/y41qZtXN6rY', 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Ren%C3%A9e_Fleming_2015.jpg/640px-Ren%C3%A9e_Fleming_2015.jpg'),
('Yo-Yo Ma', 'yoyo@cello.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 3500.00, 'Violoncelista. Acredito que a música une as pessoas. Interpretação de Bach.', 'https://www.youtube.com/embed/1prweT95Mo0', 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Yo-Yo_Ma_2011_World_Economic_Forum_Annual_Meeting.jpg/640px-Yo-Yo_Ma_2011_World_Economic_Forum_Annual_Meeting.jpg'),
('Lang Lang', 'lang@piano.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 3200.00, 'Pianista virtuoso. Técnica avançada de piano clássico e expressividade.', 'https://www.youtube.com/embed/8PTDY10tDqI', 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Lang_Lang_2024.jpg/640px-Lang_Lang_2024.jpg'),

-- POP & R&B
('Beyoncé Knowles', 'beyonce@queenb.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 5000.00, 'Vencedora de 32 Grammys. Aprenda performance vocal, dança e disciplina artística.', 'https://www.youtube.com/embed/bnVUHWCynig', 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Beyonc%C3%A9_at_The_Lion_King_European_Premiere_2019.png/640px-Beyonc%C3%A9_at_The_Lion_King_European_Premiere_2019.png'),
('Alicia Keys', 'alicia@keys.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 2000.00, 'Pianista clássica e cantora de Soul. Ensino composição (Songwriting).', 'https://www.youtube.com/embed/Urdlvw0SSEc', 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Alicia_Keys_2013.jpg/640px-Alicia_Keys_2013.jpg'),
('Ed Sheeran', 'ed@sheeran.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 1500.00, 'Compositor. Mestre em Loop Pedal e construção de músicas pop.', 'https://www.youtube.com/embed/JGwWNGJdvx8', 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Ed_Sheeran-6886_%28cropped%29.jpg/640px-Ed_Sheeran-6886_%28cropped%29.jpg'),
('Taylor Swift', 'taylor@swift.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 4000.00, 'Compositora. Storytelling através da música e conexão com os fãs.', 'https://www.youtube.com/embed/-CmadmM5cOk', 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/191125_Taylor_Swift_at_the_2019_American_Music_Awards_%28cropped%29.png/640px-191125_Taylor_Swift_at_the_2019_American_Music_Awards_%28cropped%29.png'),

-- ROCK & JAZZ
('Slash', 'slash@gnr.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 1800.00, 'Guitarrista do Guns N Roses. Aprenda riffs icônicos e solos.', 'https://www.youtube.com/embed/b36Yht8K9jU', 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Slash_live_in_London_2024.jpg/640px-Slash_live_in_London_2024.jpg'),
('Flea', 'flea@rhcp.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 1400.00, 'Baixista do Red Hot Chili Peppers. Ensino Slap Bass e Funk.', 'https://www.youtube.com/embed/raN4184yD8M', 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Flea_2016.jpg/640px-Flea_2016.jpg'),
('Dave Grohl', 'dave@foo.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 1600.00, 'Baterista (Nirvana) e Guitarrista. Energia no palco e rock.', 'https://www.youtube.com/embed/e0ppXg3JgIw', 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Dave_Grohl_2014.jpg/640px-Dave_Grohl_2014.jpg'),
('Herbie Hancock', 'herbie@jazz.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 2200.00, 'Lenda do Jazz. Improvisação, Harmonia Jazzística e Fusão.', 'https://www.youtube.com/embed/GAnyH-9l6CQ', 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Herbie_Hancock_2019.jpg/640px-Herbie_Hancock_2019.jpg'),
('John Mayer', 'john@mayer.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 1700.00, 'Mestre do Blues moderno e Pop. Técnica de guitarra e composição.', 'https://www.youtube.com/embed/32GZ3suxRn4', 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/John_Mayer_at_ByteDance_HQ.jpg/640px-John_Mayer_at_ByteDance_HQ.jpg'),

-- BRASILEIROS & OUTROS
('Toquinho', 'toquinho@mpb.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 900.00, 'Violonista. A técnica e a sensibilidade da Bossa Nova.', 'https://www.youtube.com/embed/H4D-XWkQvOQ', 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Toquinho_no_Tom_Jazz.jpg/640px-Toquinho_no_Tom_Jazz.jpg'),
('Jacob Collier', 'jacob@music.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 1300.00, 'Multi-instrumentista. Harmonia Negativa e Microtonalidade.', 'https://www.youtube.com/embed/DnB1A1d-pSY', 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Jacob_Collier_2019.jpg/640px-Jacob_Collier_2019.jpg'),
('Ivete Sangalo', 'ivete@axe.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 1000.00, 'Cantora. Como levantar multidões e técnica vocal para shows.', 'https://www.youtube.com/embed/J3y8q9R2w5c', 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Ivete_Sangalo_2016.jpg/640px-Ivete_Sangalo_2016.jpg'),
('Anitta', 'anitta@funk.com', '$2a$10$X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w/X.MwM/Jg/W.w', 1500.00, 'Cantora e Business Woman. Funk carioca, Reggaeton e gestão de carreira.', 'https://www.youtube.com/embed/scW7bH_1Z-o', 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Anitta_no_MTV_Millennial_Awards_2018_%28cropped%29.jpg/640px-Anitta_no_MTV_Millennial_Awards_2018_%28cropped%29.jpg')
ON CONFLICT (email) DO NOTHING;

-- 4. VÍNCULOS (Associações de instrumentos para filtro)
INSERT INTO professor_instrumento (professor_id, instrumento_id)
SELECT id, (SELECT id FROM instrumento WHERE nome='Voz') FROM professor WHERE email IN ('andrea@bocelli.com', 'renee@soprano.com', 'beyonce@queenb.com', 'alicia@keys.com', 'ivete@axe.com', 'jacob@music.com', 'taylor@swift.com', 'anitta@funk.com') ON CONFLICT DO NOTHING;

INSERT INTO professor_instrumento (professor_id, instrumento_id)
SELECT id, (SELECT id FROM instrumento WHERE nome='Piano') FROM professor WHERE email IN ('alicia@keys.com', 'herbie@jazz.com', 'jacob@music.com', 'andrea@bocelli.com', 'lang@piano.com') ON CONFLICT DO NOTHING;

INSERT INTO professor_instrumento (professor_id, instrumento_id)
SELECT id, (SELECT id FROM instrumento WHERE nome='Violão') FROM professor WHERE email IN ('ed@sheeran.com', 'toquinho@mpb.com', 'taylor@swift.com', 'john@mayer.com') ON CONFLICT DO NOTHING;

INSERT INTO professor_instrumento (professor_id, instrumento_id)
SELECT id, (SELECT id FROM instrumento WHERE nome='Guitarra') FROM professor WHERE email IN ('slash@gnr.com', 'dave@foo.com', 'john@mayer.com') ON CONFLICT DO NOTHING;

INSERT INTO professor_instrumento (professor_id, instrumento_id)
SELECT id, (SELECT id FROM instrumento WHERE nome='Bateria') FROM professor WHERE email IN ('dave@foo.com', 'joao.rock@gmail.com') ON CONFLICT DO NOTHING;

INSERT INTO professor_instrumento (professor_id, instrumento_id)
SELECT id, (SELECT id FROM instrumento WHERE nome='Baixo') FROM professor WHERE email='flea@rhcp.com' ON CONFLICT DO NOTHING;

INSERT INTO professor_instrumento (professor_id, instrumento_id)
SELECT id, (SELECT id FROM instrumento WHERE nome='Violoncelo') FROM professor WHERE email='yoyo@cello.com' ON CONFLICT DO NOTHING;

-- 5. CURSOS (Gigs)
INSERT INTO curso (professor_id, titulo, descricao, preco, modalidade, thumbnail_url, video_url) VALUES

-- Andrea Bocelli
((SELECT id FROM professor WHERE email='andrea@bocelli.com'), 'Masterclass: A Alma da Ópera', 'Interpretação e sentimento na música clássica.', 3000.00, 'online', 'https://images.unsplash.com/photo-1514320291840-2e0a9bf2a9ae?auto=format&fit=crop&w=400&q=80', NULL),

-- Renée Fleming
((SELECT id FROM professor WHERE email='renee@soprano.com'), 'O Canto Lírico Moderno', 'Como aplicar técnica clássica em repertórios variados.', 2800.00, 'online', 'https://images.unsplash.com/photo-1516280440614-6697288d5d38?auto=format&fit=crop&w=400&q=80', NULL),

-- Lang Lang
((SELECT id FROM professor WHERE email='lang@piano.com'), 'Piano Virtuoso', 'Exercícios de Hanon e Czerny para agilidade extrema.', 3200.00, 'online', 'https://images.unsplash.com/photo-1520523839897-bd0b52f945a0?auto=format&fit=crop&w=400&q=80', 'https://www.youtube.com/embed/8PTDY10tDqI'),

-- Slash
((SELECT id FROM professor WHERE email='slash@gnr.com'), 'Riffs e Solos de Hard Rock', 'Aprenda a criar solos memoráveis usando a escala pentatônica.', 1800.00, 'online', 'https://images.unsplash.com/photo-1550291652-6ea9114a47b1?auto=format&fit=crop&w=400&q=80', NULL),

-- Yo-Yo Ma
((SELECT id FROM professor WHERE email='yoyo@cello.com'), 'Violoncelo: Bach e Além', 'Uma jornada pelas suítes de Bach e técnica de arco.', 3500.00, 'presencial', 'https://images.unsplash.com/photo-1560243972-4627e0022067?auto=format&fit=crop&w=400&q=80', NULL),

-- Alicia Keys
((SELECT id FROM professor WHERE email='alicia@keys.com'), 'Composição e Piano Soul', 'Escreva músicas que conectam e acompanhe-se no piano.', 2000.00, 'online', 'https://images.unsplash.com/photo-1520523839897-bd0b52f945a0?auto=format&fit=crop&w=400&q=80', NULL),

-- Ed Sheeran
((SELECT id FROM professor WHERE email='ed@sheeran.com'), 'Dominando o Loop Pedal', 'Crie arranjos completos ao vivo apenas com violão e voz.', 1500.00, 'online', 'https://images.unsplash.com/photo-1525201548942-d8732f6617a0?auto=format&fit=crop&w=400&q=80', NULL),

-- Taylor Swift
((SELECT id FROM professor WHERE email='taylor@swift.com'), 'Storytelling na Música', 'Como escrever letras que contam histórias e emocionam.', 4000.00, 'online', 'https://images.unsplash.com/photo-1493225255756-d9584f8606e9?auto=format&fit=crop&w=400&q=80', NULL),

-- John Mayer
((SELECT id FROM professor WHERE email='john@mayer.com'), 'Blues Guitar Moderno', 'Tríades, Hendrix-style e improvisação melódica.', 1700.00, 'online', 'https://images.unsplash.com/photo-1550291652-6ea9114a47b1?auto=format&fit=crop&w=400&q=80', NULL),

-- Flea
((SELECT id FROM professor WHERE email='flea@rhcp.com'), 'Baixo Funk e Slap', 'Aumente seu groove e aprenda a técnica de slap agressiva.', 1400.00, 'online', 'https://images.unsplash.com/photo-1460039230329-eb070fc6c77c?auto=format&fit=crop&w=400&q=80', NULL),

-- Dave Grohl
((SELECT id FROM professor WHERE email='dave@foo.com'), 'Bateria Rock: Energia Pura', 'Toque com força e precisão. Técnicas de rock alternativo.', 1600.00, 'presencial', 'https://images.unsplash.com/photo-1519892300165-cb5542fb47c7?auto=format&fit=crop&w=400&q=80', NULL),

-- Herbie Hancock
((SELECT id FROM professor WHERE email='herbie@jazz.com'), 'Improvisação no Jazz', 'Teoria avançada de harmonia e liberdade no piano.', 2200.00, 'online', 'https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?auto=format&fit=crop&w=400&q=80', NULL),

-- Toquinho
((SELECT id FROM professor WHERE email='toquinho@mpb.com'), 'Violão Bossa Nova', 'Acordes, batidas e a harmonia da música brasileira.', 900.00, 'online', 'https://images.unsplash.com/photo-1514117445516-2ecfc9c4ec90?auto=format&fit=crop&w=400&q=80', NULL),

-- Jacob Collier
((SELECT id FROM professor WHERE email='jacob@music.com'), 'Teoria Musical Avançada', 'Harmonia negativa, microtonalidade e ritmos complexos.', 1300.00, 'online', 'https://images.unsplash.com/photo-1507838153414-b4b713384ebd?auto=format&fit=crop&w=400&q=80', NULL),

-- Anitta
((SELECT id FROM professor WHERE email='anitta@funk.com'), 'Carreira e Performance Pop', 'Dança, presença de palco e gestão de carreira internacional.', 1500.00, 'online', 'https://images.unsplash.com/photo-1516280440614-6697288d5d38?auto=format&fit=crop&w=400&q=80', NULL),

-- ORIGINAIS
((SELECT id FROM professor WHERE email='daniel.augusto@gmail.com'), 'Violão Clássico: Do Zero ao Avançado', 'Técnica completa.', 120.00, 'online', 'https://images.unsplash.com/photo-1605020420620-20c943cc4669?auto=format&fit=crop&w=400&q=80', 'https://www.youtube.com/embed/2t7s5FZZwXo'),
((SELECT id FROM professor WHERE email='mel.peixoto@gmail.com'), 'Canto Lírico: Técnica Bel Canto', 'Ópera e saúde vocal.', 150.00, 'presencial', 'https://images.unsplash.com/photo-1516280440614-6697288d5d38?auto=format&fit=crop&w=400&q=80', 'https://www.youtube.com/embed/a7zhAlhABrE'),
((SELECT id FROM professor WHERE email='joao.rock@gmail.com'), 'Bateria Rock: Primeiro Show', 'Levadas essenciais.', 90.00, 'presencial', 'https://images.unsplash.com/photo-1533174072545-e8d4aa97edf9?auto=format&fit=crop&w=400&q=80', 'https://www.youtube.com/embed/I1kl_PZ-gT0')
ON CONFLICT DO NOTHING;

-- 6. DISPONIBILIDADE (Para todos)
INSERT INTO disponibilidade (professor_id, dia_semana, hora_inicio, hora_fim) 
SELECT id, 1, '10:00', '18:00' FROM professor ON CONFLICT DO NOTHING;
/*
Navicat MySQL Data Transfer

Source Server         : dbem
Source Server Version : 80018
Source Host           : localhost:3306
Source Database       : booktop250

Target Server Type    : MYSQL
Target Server Version : 80018
File Encoding         : 65001

Date: 2020-07-27 01:26:37
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for bookinfo
-- ----------------------------
DROP TABLE IF EXISTS `bookinfo`;
CREATE TABLE `bookinfo` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `book_name` varchar(45) DEFAULT NULL,
  `english_name` varchar(45) DEFAULT NULL,
  `author` varchar(45) DEFAULT NULL,
  `img` varchar(100) DEFAULT NULL,
  `publish` varchar(45) DEFAULT NULL,
  `price` varchar(45) DEFAULT NULL,
  `score` float DEFAULT '0',
  `description` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=25 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of bookinfo
-- ----------------------------
INSERT INTO `bookinfo` VALUES ('1', '红楼梦', '', '[清] 曹雪芹 著 ', 'https://img1.doubanio.com/view/subject/s/public/s1070959.jpg', ' 人民文学出版社 ', ' 59.70元', '9.6', '都云作者痴，谁解其中味？');
INSERT INTO `bookinfo` VALUES ('2', '活着', '', '余华 ', 'https://img3.doubanio.com/view/subject/s/public/s29053580.jpg', ' 作家出版社 ', ' 20.00元', '9.4', '生的苦难与伟大');
INSERT INTO `bookinfo` VALUES ('3', '百年孤独', 'Cien años de soledad', '[哥伦比亚] 加西亚·马尔克斯 ', 'https://img9.doubanio.com/view/subject/s/public/s6384944.jpg', ' 南海出版公司 ', ' 39.50元', '9.2', '魔幻现实主义文学代表作');
INSERT INTO `bookinfo` VALUES ('4', '1984', 'Nineteen Eighty-Four', '[英] 乔治·奥威尔 ', 'https://img1.doubanio.com/view/subject/s/public/s4371408.jpg', ' 北京十月文艺出版社 ', ' 28.00', '9.3', '栗树荫下，我出卖你，你出卖我');
INSERT INTO `bookinfo` VALUES ('5', '飘', 'Gone with the Wind', '[美国] 玛格丽特·米切尔 ', 'https://img1.doubanio.com/view/subject/s/public/s1078958.jpg', ' 译林出版社 ', ' 40.00元', '9.3', '革命时期的爱情，随风而逝');
INSERT INTO `bookinfo` VALUES ('6', '三体全集', '', '刘慈欣 ', 'https://img3.doubanio.com/view/subject/s/public/s30016152.jpg', ' 重庆出版社 ', ' 168.00元', '9.4', '地球往事三部曲');
INSERT INTO `bookinfo` VALUES ('7', '三国演义（全二册）', '', '[明] 罗贯中 ', 'https://img3.doubanio.com/view/subject/s/public/s1076932.jpg', ' 人民文学出版社 ', ' 39.50元', '9.3', '是非成败转头空');
INSERT INTO `bookinfo` VALUES ('8', '白夜行', '', '[日] 东野圭吾 ', 'https://img1.doubanio.com/view/subject/s/public/s24514468.jpg', ' 南海出版公司 ', ' 39.50元', '9.1', '一宗离奇命案牵出跨度近20年步步惊心的故事');
INSERT INTO `bookinfo` VALUES ('9', '福尔摩斯探案全集（上中下）', '', '[英] 阿·柯南道尔 ', 'https://img3.doubanio.com/view/subject/s/public/s1229240.jpg', ' 1981-8 ', '68.00元', '9.2', '名侦探的代名词');
INSERT INTO `bookinfo` VALUES ('10', '房思琪的初恋乐园', '', '林奕含 ', 'https://img3.doubanio.com/view/subject/s/public/s29651121.jpg', ' 北京联合出版公司 ', ' 45.00元', '9.2', '向死而生的文学绝唱');
INSERT INTO `bookinfo` VALUES ('11', '动物农场', 'Animal Farm', '[英] 乔治·奥威尔 ', 'https://img3.doubanio.com/view/subject/s/public/s2347590.jpg', ' 上海译文出版社 ', ' 10.00元', '9.2', '太阳底下并无新事');
INSERT INTO `bookinfo` VALUES ('12', '小王子', 'Le Petit Prince', '[法] 圣埃克苏佩里 ', 'https://img3.doubanio.com/view/subject/s/public/s1103152.jpg', ' 人民文学出版社 ', ' 22.00元', '9', '献给长成了大人的孩子们');
INSERT INTO `bookinfo` VALUES ('13', '天龙八部', '', '金庸 ', 'https://img1.doubanio.com/view/subject/s/public/s23632058.jpg', ' 生活.读书.新知三联书店 ', ' 96.0', '9.1', '有情皆孽，无人不冤');
INSERT INTO `bookinfo` VALUES ('14', '撒哈拉的故事', '', '三毛 ', 'https://img3.doubanio.com/view/subject/s/public/s1066570.jpg', ' 哈尔滨出版社 ', ' 15.80元', '9.2', '游荡的自由灵魂');
INSERT INTO `bookinfo` VALUES ('15', '安徒生童话故事集', '', '（丹麦）安徒生 ', 'https://img3.doubanio.com/view/subject/s/public/s1034062.jpg', ' 人民文学出版社 ', ' 25.00元', '9.2', '为了争取未来的一代');
INSERT INTO `bookinfo` VALUES ('16', '哈利•波特', 'Harry Potter', 'J.K.罗琳 (J.K.Rowling) ', 'https://img9.doubanio.com/view/subject/s/public/s29552296.jpg', ' 人民文学出版社 ', ' 498.00元', '9.7', '从9¾站台开始的旅程');
INSERT INTO `bookinfo` VALUES ('17', '沉默的大多数', '', '王小波 ', 'https://img1.doubanio.com/view/subject/s/public/s1447349.jpg', ' 中国青年出版社 ', ' 27.00元', '9.1', '沉默是沉默者的通行证');
INSERT INTO `bookinfo` VALUES ('18', '人类简史', 'A brief history of humankind', '[以色列] 尤瓦尔·赫拉利 ', 'https://img3.doubanio.com/view/subject/s/public/s27814883.jpg', ' 中信出版社 ', ' CNY 68.00', '9.1', '跟着人类一同走过十万年');
INSERT INTO `bookinfo` VALUES ('19', '围城', '', '钱锺书 ', 'https://img3.doubanio.com/view/subject/s/public/s1070222.jpg', ' 人民文学出版社 ', ' 19.00', '8.9', '幽默的语言和对生活深刻的观察');
INSERT INTO `bookinfo` VALUES ('20', '平凡的世界（全三部）', '', '路遥 ', 'https://img3.doubanio.com/view/subject/s/public/s1144911.jpg', ' 人民文学出版社 ', ' 64.00元', '9', '中国当代城乡生活全景');
INSERT INTO `bookinfo` VALUES ('21', '霍乱时期的爱情', 'El amor en los tiempos del cólera', '[哥伦比亚] 加西亚·马尔克斯 ', 'https://img3.doubanio.com/view/subject/s/public/s11284102.jpg', ' 南海出版公司 ', ' 39.50元', '9', '义无反顾地直达爱情的核心');
INSERT INTO `bookinfo` VALUES ('22', '明朝那些事儿（1-9）', '', '当年明月 ', 'https://img9.doubanio.com/view/subject/s/public/s3745215.jpg', ' 中国海关出版社 ', ' 358.20元', '9.1', '不拘一格的历史书写');
INSERT INTO `bookinfo` VALUES ('23', '杀死一只知更鸟', 'To Kill a Mocking Bird', '[美] 哈珀·李 ', 'https://img3.doubanio.com/view/subject/s/public/s23128183.jpg', ' 译林出版社 ', ' 32.00元', '9.2', '有一种东西不能遵循从众原则，那就是——人的良心');
INSERT INTO `bookinfo` VALUES ('24', '笑傲江湖（全四册）', '', '金庸 ', 'https://img9.doubanio.com/view/subject/s/public/s2157335.jpg', ' 生活·读书·新知三联书店 ', ' 76.80元', '9', '欲练此功，必先自宫');
